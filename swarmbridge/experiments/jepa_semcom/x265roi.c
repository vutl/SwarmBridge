/* B-E8 (review #34): minimal libx265 adapter with per-frame, per-16x16-block QP offsets (x265_picture.quantOffsets).
 *
 *   x265roi W H FPS CRF QOFF OUT [key=value ...]
 *
 * stdin : raw yuv420p frames (W x H, 8 bit), e.g. from `ffmpeg ... -pix_fmt yuv420p -f rawvideo -`.
 * QOFF  : '-' (no offsets: plain CRF encoding) or a file of float32 offsets, ceil(W/16) * ceil(H/16) per frame in
 *         raster order, one block row after another, frames in order. Offsets are added by x265 on top of its rate
 *         control / adaptive-quantisation decisions (aq-mode must be > 0; it is in the presets used).
 * OUT   : raw Annex-B HEVC stream (parameter sets once, written from x265_encoder_headers; no repeated headers).
 * extra : x265 parameters, parsed with x265_param_parse (the same names as --x265-params in ffmpeg).
 * Base setting (same as B-E7 b3_video.enc_cmd): preset slow, tune zerolatency (no B-frames, no lookahead, no
 * frame-parallelism), and the caller passes keyint=-1 min-keyint=1 scenecut=0 bframes=0 rc-lookahead=0 info=0.
 * The offsets of frame t are read before frame t is submitted and only frame t is submitted: the encoder never sees
 * a later frame or a later offset map when it codes frame t (zerolatency: one frame in, its packet out).
 * Per-frame NAL bytes are written to stderr as "pkt <frame> <bytes>" lines (headers: "hdr <bytes>").
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <x265.h>

static size_t write_nals(FILE *f, x265_nal *nals, uint32_t n) {
    size_t s = 0;
    for (uint32_t i = 0; i < n; i++) {
        fwrite(nals[i].payload, 1, nals[i].sizeBytes, f);
        s += nals[i].sizeBytes;
    }
    return s;
}

int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr, "usage: x265roi W H FPS CRF QOFF|- OUT [key=value ...]\n");
        return 1;
    }
    int W = atoi(argv[1]), H = atoi(argv[2]), fps = atoi(argv[3]);
    const char *qpath = argv[5], *opath = argv[6];
    x265_param *p = x265_param_alloc();
    if (x265_param_default_preset(p, "slow", "zerolatency") < 0) return 2;
    p->sourceWidth = W;
    p->sourceHeight = H;
    p->fpsNum = fps;
    p->fpsDenom = 1;
    p->internalCsp = X265_CSP_I420;
    p->bRepeatHeaders = 0;
    if (x265_param_parse(p, "crf", argv[4])) return 3;
    for (int i = 7; i < argc; i++) {
        char *kv = strdup(argv[i]);
        char *eq = strchr(kv, '=');
        if (!eq) return 3;
        *eq = 0;
        if (x265_param_parse(p, kv, eq + 1)) {
            fprintf(stderr, "bad x265 parameter %s\n", argv[i]);
            return 3;
        }
        free(kv);
    }
    if (p->rc.aqMode == 0 && strcmp(qpath, "-")) {
        fprintf(stderr, "quant offsets need aq-mode > 0\n");
        return 4;
    }
    if (p->rc.qgSize == 8) {
        fprintf(stderr, "qg-size 8 not supported by this adapter\n");
        return 4;
    }
    x265_encoder *enc = x265_encoder_open(p);
    if (!enc) return 5;
    FILE *fo = fopen(opath, "wb");
    FILE *fq = strcmp(qpath, "-") ? fopen(qpath, "rb") : NULL;
    if (!fo || (strcmp(qpath, "-") && !fq)) return 6;
    x265_nal *nals;
    uint32_t nn;
    if (x265_encoder_headers(enc, &nals, &nn) < 0) return 7;
    fprintf(stderr, "hdr %zu\n", write_nals(fo, nals, nn));
    int nb = ((W + 15) / 16) * ((H + 15) / 16);
    float *q = malloc(sizeof(float) * nb);
    size_t ysz = (size_t)W * H, csz = (size_t)(W / 2) * (H / 2), fsz = ysz + 2 * csz;
    uint8_t *buf = malloc(fsz);
    x265_picture *pic = x265_picture_alloc(), *pout = x265_picture_alloc();
    x265_picture_init(p, pic);
    pic->planes[0] = buf;
    pic->planes[1] = buf + ysz;
    pic->planes[2] = buf + ysz + csz;
    pic->stride[0] = W;
    pic->stride[1] = pic->stride[2] = W / 2;
    int t = 0, outn = 0;
    while (fread(buf, 1, fsz, stdin) == fsz) {
        pic->pts = t;
        pic->quantOffsets = NULL;
        if (fq) {
            if (fread(q, sizeof(float), nb, fq) != (size_t)nb) {
                fprintf(stderr, "offset file too short at frame %d\n", t);
                return 8;
            }
            pic->quantOffsets = q;            /* copied by x265 inside x265_encoder_encode */
        }
        int r = x265_encoder_encode(enc, &nals, &nn, pic, pout);
        if (r < 0) return 9;
        if (r > 0) fprintf(stderr, "pkt %d %zu\n", outn++, write_nals(fo, nals, nn));
        t++;
    }
    while (x265_encoder_encode(enc, &nals, &nn, NULL, pout) > 0)
        fprintf(stderr, "pkt %d %zu\n", outn++, write_nals(fo, nals, nn));
    fprintf(stderr, "frames %d packets %d\n", t, outn);
    x265_encoder_close(enc);
    x265_picture_free(pic);
    x265_picture_free(pout);
    x265_param_free(p);
    fclose(fo);
    if (fq) fclose(fq);
    free(q);
    free(buf);
    return outn == t ? 0 : 10;
}
