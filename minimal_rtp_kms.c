#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <rockchip/rk_mpi.h>
#include <xf86drm.h>
#include <xf86drmMode.h>
#include <drm_fourcc.h>

#include "drm.h"

#define READ_BUF_SIZE (1024 * 1024)
#define MAX_FRAMES 24

static const char *pipeline_template =
    "udpsrc port=%d caps=\"application/x-rtp, media=(string)video, encoding-name=(string)H265, clock-rate=(int)90000\" ! "
    "rtph265depay ! "
    "h265parse config-interval=-1 ! "
    "video/x-h265, stream-format=\"byte-stream\" ! "
    "appsink drop=true name=out_appsink";

static volatile sig_atomic_t running = 1;
static int frm_eos = 0;
static int drm_fd = -1;
static struct modeset_output *output_list = NULL;
static int video_zpos = 1;

static pthread_mutex_t video_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t video_cond = PTHREAD_COND_INITIALIZER;
static uint32_t pending_fb = 0;
static uint64_t pending_pts = 0;

static uint8_t *nal_buffer = NULL;
static MppPacket mpp_packet_handle;

static struct {
    MppCtx ctx;
    MppApi *mpi;
    MppBufferGroup frm_grp;
    struct {
        int prime_fd;
        uint32_t fb_id;
        uint32_t handle;
    } frame_to_drm[MAX_FRAMES];
} decoder;

static uint64_t get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000ULL + ts.tv_nsec / 1000000ULL;
}

static void handle_signal(int signum) {
    (void)signum;
    running = 0;
}

static void set_control_verbose(MppApi *mpi, MppCtx ctx, MpiCmd control, RK_U32 enable) {
    RK_U32 res = mpi->control(ctx, control, &enable);
    if (res) {
        fprintf(stderr, "Could not set control %d -> %u (%d)\n", control, enable, res);
    }
}

static void set_mpp_decoding_parameters(MppApi *mpi, MppCtx ctx) {
    MppDecCfg cfg = NULL;
    int ret = mpp_dec_cfg_init(&cfg);
    if (ret) {
        fprintf(stderr, "mpp_dec_cfg_init failed: %d\n", ret);
        return;
    }

    ret = mpi->control(ctx, MPP_DEC_GET_CFG, cfg);
    if (ret) {
        fprintf(stderr, "MPP_DEC_GET_CFG failed: %d\n", ret);
        goto out;
    }

    ret = mpp_dec_cfg_set_u32(cfg, "base:split_parse", 1);
    if (ret) {
        fprintf(stderr, "Failed to enable split_parse: %d\n", ret);
        goto out;
    }

    ret = mpi->control(ctx, MPP_DEC_SET_CFG, cfg);
    if (ret) {
        fprintf(stderr, "MPP_DEC_SET_CFG failed: %d\n", ret);
    }

    set_control_verbose(mpi, ctx, MPP_DEC_SET_DISABLE_ERROR, 0xffff);
    set_control_verbose(mpi, ctx, MPP_DEC_SET_IMMEDIATE_OUT, 0xffff);
    set_control_verbose(mpi, ctx, MPP_DEC_SET_ENABLE_FAST_PLAY, 0xffff);
    set_control_verbose(mpi, ctx, MPP_DEC_SET_PARSER_SPLIT_MODE, 0);

out:
    if (cfg) {
        mpp_dec_cfg_deinit(cfg);
    }
}

static void reset_frame_map(void) {
    for (int i = 0; i < MAX_FRAMES; ++i) {
        decoder.frame_to_drm[i].prime_fd = -1;
        decoder.frame_to_drm[i].fb_id = 0;
        decoder.frame_to_drm[i].handle = 0;
    }
}

static void release_frame_group(void) {
    if (!decoder.frm_grp) {
        return;
    }

    for (int i = 0; i < MAX_FRAMES; ++i) {
        if (decoder.frame_to_drm[i].fb_id) {
            drmModeRmFB(drm_fd, decoder.frame_to_drm[i].fb_id);
            decoder.frame_to_drm[i].fb_id = 0;
        }
        if (decoder.frame_to_drm[i].prime_fd >= 0) {
            close(decoder.frame_to_drm[i].prime_fd);
            decoder.frame_to_drm[i].prime_fd = -1;
        }
        if (decoder.frame_to_drm[i].handle) {
            struct drm_mode_destroy_dumb dmd = {
                .handle = decoder.frame_to_drm[i].handle,
            };
            ioctl(drm_fd, DRM_IOCTL_MODE_DESTROY_DUMB, &dmd);
            decoder.frame_to_drm[i].handle = 0;
        }
    }

    mpp_buffer_group_clear(decoder.frm_grp);
    mpp_buffer_group_put(decoder.frm_grp);
    decoder.frm_grp = NULL;
}

static void init_buffer_from_frame(MppFrame frame) {
    RK_U32 width = mpp_frame_get_width(frame);
    RK_U32 height = mpp_frame_get_height(frame);
    RK_U32 hor_stride = mpp_frame_get_hor_stride(frame);
    RK_U32 ver_stride = mpp_frame_get_ver_stride(frame);
    MppFrameFormat fmt = mpp_frame_get_fmt(frame);

    if (fmt != MPP_FMT_YUV420SP && fmt != MPP_FMT_YUV420SP_10BIT) {
        fprintf(stderr, "Unexpected format %d\n", fmt);
        return;
    }

    output_list->video_frm_width = width;
    output_list->video_frm_height = height;
    output_list->video_fb_x = 0;
    output_list->video_fb_y = 0;
    output_list->video_fb_width = output_list->mode.hdisplay;
    output_list->video_fb_height = output_list->mode.vdisplay;

    release_frame_group();

    int ret = mpp_buffer_group_get_external(&decoder.frm_grp, MPP_BUFFER_TYPE_DRM);
    if (ret) {
        fprintf(stderr, "Failed to get external buffer group: %d\n", ret);
        return;
    }

    for (int i = 0; i < MAX_FRAMES; ++i) {
        struct drm_mode_create_dumb dmcd;
        memset(&dmcd, 0, sizeof(dmcd));
        dmcd.bpp = (fmt == MPP_FMT_YUV420SP) ? 8 : 10;
        dmcd.width = hor_stride;
        dmcd.height = ver_stride * 2;

        do {
            ret = ioctl(drm_fd, DRM_IOCTL_MODE_CREATE_DUMB, &dmcd);
        } while (ret == -1 && (errno == EINTR || errno == EAGAIN));
        if (ret) {
            fprintf(stderr, "DRM_IOCTL_MODE_CREATE_DUMB failed: %d (%s)\n", ret, strerror(errno));
            continue;
        }
        decoder.frame_to_drm[i].handle = dmcd.handle;

        struct drm_prime_handle dph;
        memset(&dph, 0, sizeof(dph));
        dph.handle = dmcd.handle;
        dph.fd = -1;
        do {
            ret = ioctl(drm_fd, DRM_IOCTL_PRIME_HANDLE_TO_FD, &dph);
        } while (ret == -1 && (errno == EINTR || errno == EAGAIN));
        if (ret) {
            fprintf(stderr, "PRIME_HANDLE_TO_FD failed: %d (%s)\n", ret, strerror(errno));
            continue;
        }

        MppBufferInfo info;
        memset(&info, 0, sizeof(info));
        info.type = MPP_BUFFER_TYPE_DRM;
        info.size = dmcd.width * dmcd.height;
        info.fd = dph.fd;
        ret = mpp_buffer_commit(decoder.frm_grp, &info);
        if (ret) {
            fprintf(stderr, "mpp_buffer_commit failed: %d\n", ret);
            close(dph.fd);
            continue;
        }

        decoder.frame_to_drm[i].prime_fd = info.fd;
        if (dph.fd != info.fd) {
            close(dph.fd);
        }

        uint32_t handles[4] = {0};
        uint32_t pitches[4] = {0};
        uint32_t offsets[4] = {0};
        handles[0] = decoder.frame_to_drm[i].handle;
        handles[1] = decoder.frame_to_drm[i].handle;
        pitches[0] = hor_stride;
        pitches[1] = hor_stride;
        offsets[0] = 0;
        offsets[1] = pitches[0] * ver_stride;

        ret = drmModeAddFB2(drm_fd, width, height, DRM_FORMAT_NV12,
                            handles, pitches, offsets, &decoder.frame_to_drm[i].fb_id, 0);
        if (ret) {
            fprintf(stderr, "drmModeAddFB2 failed: %d (%s)\n", ret, strerror(errno));
        }
    }

    decoder.mpi->control(decoder.ctx, MPP_DEC_SET_EXT_BUF_GROUP, decoder.frm_grp);
    decoder.mpi->control(decoder.ctx, MPP_DEC_SET_INFO_CHANGE_READY, NULL);

    modeset_perform_modeset(drm_fd, output_list, output_list->video_request,
                            &output_list->video_plane,
                            decoder.frame_to_drm[0].fb_id,
                            output_list->video_frm_width,
                            output_list->video_frm_height,
                            video_zpos);
}

static int feed_packet_to_decoder(const uint8_t *data, size_t length) {
    if (!length) {
        return -1;
    }

    mpp_packet_set_length(mpp_packet_handle, 0);
    mpp_packet_set_size(mpp_packet_handle, READ_BUF_SIZE);
    mpp_packet_set_data(mpp_packet_handle, (void *)data);
    mpp_packet_set_pos(mpp_packet_handle, (void *)data);
    mpp_packet_set_length(mpp_packet_handle, length);
    mpp_packet_set_pts(mpp_packet_handle, (RK_S64)get_time_ms());

    while (running) {
        int ret = decoder.mpi->decode_put_packet(decoder.ctx, mpp_packet_handle);
        if (ret == MPP_OK) {
            return 0;
        }
        usleep(2000);
    }
    return -1;
}

static void send_decoder_eos(void) {
    if (!decoder.mpi || !decoder.ctx) {
        return;
    }

    mpp_packet_set_length(mpp_packet_handle, 0);
    mpp_packet_set_size(mpp_packet_handle, READ_BUF_SIZE);
    mpp_packet_set_data(mpp_packet_handle, nal_buffer);
    mpp_packet_set_pos(mpp_packet_handle, nal_buffer);
    mpp_packet_set_eos(mpp_packet_handle);

    while (decoder.mpi->decode_put_packet(decoder.ctx, mpp_packet_handle) != MPP_OK) {
        usleep(2000);
    }
}

static void *frame_thread(void *arg) {
    (void)arg;
    while (running && !frm_eos) {
        MppFrame frame = NULL;
        int ret = decoder.mpi->decode_get_frame(decoder.ctx, &frame);
        if (ret || !frame) {
            continue;
        }

        if (mpp_frame_get_info_change(frame)) {
            init_buffer_from_frame(frame);
        } else {
            MppBuffer buffer = mpp_frame_get_buffer(frame);
            if (buffer) {
                MppBufferInfo info;
                memset(&info, 0, sizeof(info));
                ret = mpp_buffer_info_get(buffer, &info);
                if (!ret) {
                    for (int i = 0; i < MAX_FRAMES; ++i) {
                        if (decoder.frame_to_drm[i].prime_fd == info.fd) {
                            pthread_mutex_lock(&video_mutex);
                            pending_fb = decoder.frame_to_drm[i].fb_id;
                            pending_pts = mpp_frame_get_pts(frame);
                            pthread_cond_signal(&video_cond);
                            pthread_mutex_unlock(&video_mutex);
                            break;
                        }
                    }
                }
            }
        }

        frm_eos = mpp_frame_get_eos(frame);
        mpp_frame_deinit(&frame);
    }

    running = 0;
    pthread_cond_signal(&video_cond);
    return NULL;
}

static void *display_thread(void *arg) {
    (void)arg;
    while (running) {
        pthread_mutex_lock(&video_mutex);
        while (running && pending_fb == 0) {
            pthread_cond_wait(&video_cond, &video_mutex);
        }
        uint32_t fb = pending_fb;
        pending_fb = 0;
        pthread_mutex_unlock(&video_mutex);

        if (!running) {
            break;
        }

        if (fb) {
            if (output_list->video_request) {
                drmModeAtomicFree(output_list->video_request);
            }
            output_list->video_request = drmModeAtomicAlloc();
            set_drm_object_property(output_list->video_request,
                                    &output_list->video_plane,
                                    "FB_ID", fb);
            drmModeAtomicCommit(drm_fd, output_list->video_request,
                                DRM_MODE_ATOMIC_NONBLOCK, NULL);
        }
    }
    return NULL;
}

static void cleanup(void) {
    running = 0;
    pthread_cond_signal(&video_cond);

    if (decoder.mpi && decoder.ctx) {
        decoder.mpi->reset(decoder.ctx);
        mpp_destroy(decoder.ctx);
        decoder.ctx = NULL;
        decoder.mpi = NULL;
    }

    release_frame_group();

    if (mpp_packet_handle) {
        mpp_packet_deinit(&mpp_packet_handle);
        mpp_packet_handle = NULL;
    }
    free(nal_buffer);
    nal_buffer = NULL;

    if (output_list) {
        modeset_cleanup(drm_fd, output_list);
        output_list = NULL;
    }

    if (drm_fd >= 0) {
        close(drm_fd);
        drm_fd = -1;
    }
}

int main(int argc, char **argv) {
    int port = 5600;
    const char *drm_node = "/dev/dri/card0";
    uint16_t mode_width = 1920;
    uint16_t mode_height = 1080;
    uint32_t mode_vrefresh = 60;
    uint32_t plane_override = 0;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--port") && i + 1 < argc) {
            port = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--drm-node") && i + 1 < argc) {
            drm_node = argv[++i];
        } else if (!strcmp(argv[i], "--plane-id") && i + 1 < argc) {
            plane_override = (uint32_t)strtoul(argv[++i], NULL, 0);
        } else if (!strcmp(argv[i], "--mode") && i + 1 < argc) {
            const char *mode = argv[++i];
            const char *at = strchr(mode, '@');
            if (at) {
                mode_vrefresh = (uint32_t)atoi(at + 1);
            }
            mode_height = (uint16_t)atoi(strchr(mode, 'x') + 1);
            mode_width = (uint16_t)atoi(mode);
        } else if (!strcmp(argv[i], "--help")) {
            printf("Usage: %s [--port N] [--drm-node PATH] [--plane-id ID] [--mode WxH@Hz]\n", argv[0]);
            return 0;
        }
    }

    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    gst_init(&argc, &argv);

    int ret = modeset_open(&drm_fd, drm_node);
    if (ret < 0) {
        fprintf(stderr, "Failed to open DRM node %s (%d)\n", drm_node, ret);
        return 1;
    }

    output_list = modeset_prepare(drm_fd, mode_width, mode_height, mode_vrefresh,
                                  plane_override, 0);
    if (!output_list) {
        fprintf(stderr, "modeset_prepare failed\n");
        cleanup();
        return 1;
    }

    reset_frame_map();

    nal_buffer = malloc(READ_BUF_SIZE);
    if (!nal_buffer) {
        fprintf(stderr, "Failed to allocate NAL buffer\n");
        cleanup();
        return 1;
    }

    ret = mpp_packet_init(&mpp_packet_handle, nal_buffer, READ_BUF_SIZE);
    if (ret) {
        fprintf(stderr, "mpp_packet_init failed: %d\n", ret);
        cleanup();
        return 1;
    }

    MppCodingType mpp_type = MPP_VIDEO_CodingHEVC;
    ret = mpp_create(&decoder.ctx, &decoder.mpi);
    if (ret) {
        fprintf(stderr, "mpp_create failed: %d\n", ret);
        cleanup();
        return 1;
    }

    set_mpp_decoding_parameters(decoder.mpi, decoder.ctx);

    ret = mpp_init(decoder.ctx, MPP_CTX_DEC, mpp_type);
    if (ret) {
        fprintf(stderr, "mpp_init failed: %d\n", ret);
        cleanup();
        return 1;
    }

    set_mpp_decoding_parameters(decoder.mpi, decoder.ctx);

    int block_param = MPP_POLL_BLOCK;
    decoder.mpi->control(decoder.ctx, MPP_SET_OUTPUT_BLOCK, &block_param);

    pthread_t frame_tid;
    pthread_t display_tid;
    pthread_create(&frame_tid, NULL, frame_thread, NULL);
    pthread_create(&display_tid, NULL, display_thread, NULL);

    char pipeline_desc[512];
    snprintf(pipeline_desc, sizeof(pipeline_desc), pipeline_template, port);

    GError *error = NULL;
    GstElement *pipeline = gst_parse_launch(pipeline_desc, &error);
    if (!pipeline || error) {
        fprintf(stderr, "Pipeline creation failed: %s\n", error ? error->message : "unknown");
        if (error) {
            g_error_free(error);
        }
        running = 0;
    }

    GstElement *appsink = NULL;
    if (pipeline) {
        appsink = gst_bin_get_by_name(GST_BIN(pipeline), "out_appsink");
        if (!appsink) {
            fprintf(stderr, "Failed to find appsink\n");
            running = 0;
        }
    }

    if (pipeline && appsink) {
        gst_element_set_state(pipeline, GST_STATE_PLAYING);

        while (running) {
            GstSample *sample = gst_app_sink_try_pull_sample(GST_APP_SINK(appsink),
                                                             100 * GST_MSECOND);
            if (!sample) {
                continue;
            }

            GstBuffer *buffer = gst_sample_get_buffer(sample);
            if (buffer) {
                GstMapInfo map;
                if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
                    if (map.size <= READ_BUF_SIZE) {
                        memcpy(nal_buffer, map.data, map.size);
                        feed_packet_to_decoder(nal_buffer, map.size);
                    } else {
                        fprintf(stderr, "Sample too large (%zu bytes)\n", map.size);
                    }
                    gst_buffer_unmap(buffer, &map);
                }
            }
            gst_sample_unref(sample);
        }

        gst_element_send_event(pipeline, gst_event_new_eos());
        gst_element_set_state(pipeline, GST_STATE_NULL);
    }

    send_decoder_eos();

    if (appsink) {
        gst_object_unref(appsink);
    }
    if (pipeline) {
        gst_object_unref(pipeline);
    }

    running = 0;
    pthread_mutex_lock(&video_mutex);
    pthread_cond_broadcast(&video_cond);
    pthread_mutex_unlock(&video_mutex);

    pthread_join(frame_tid, NULL);
    pthread_join(display_tid, NULL);

    cleanup();
    return 0;
}
