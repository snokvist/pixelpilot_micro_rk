gcc src/minimal_rtp_kms.c src/drm.c       $(pkg-config --cflags --libs gstreamer-app-1.0 gstreamer-1.0 cairo libdrm)       -lrockchip_mpp -lpthread       -o minimal_rtp_kms
