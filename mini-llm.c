/*
 * mini-llm.c - Single file LLM inference engine
 * Build: gcc -O3 -o mini-llm mini-llm.c -lm -lpthread
 * Usage: ./mini-llm model.gguf
 * Then:  curl http://localhost:8080/api/generate -d '{"prompt":"Hello"}'
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#define HAS_NEON 1
#else
#define HAS_NEON 0
#endif

/* ============================================================
 * GGUF Parser
 * ============================================================ */

#define GGUF_MAGIC 0x46554747
#define GGUF_VERSION 3

enum gguf_type {
  GGUF_TYPE_UINT8    = 0,
  GGUF_TYPE_INT8     = 1,
  GGUF_TYPE_UINT16   = 2,
  GGUF_TYPE_INT16    = 3,
  GGUF_TYPE_UINT32   = 4,
  GGUF_TYPE_INT32    = 5,
  GGUF_TYPE_FLOAT32  = 6,
  GGUF_TYPE_BOOL     = 7,
  GGUF_TYPE_STRING   = 8,
  GGUF_TYPE_ARRAY    = 9,
  GGUF_TYPE_UINT64   = 10,
  GGUF_TYPE_INT64    = 11,
  GGUF_TYPE_FLOAT64  = 12
};

enum gguf_tensor_type {
  GGUF_F32     = 0,
  GGUF_F16     = 1,
  GGUF_Q4_0    = 2,
  GGUF_Q4_1    = 3,
  GGUF_Q5_0    = 6,
  GGUF_Q5_1    = 7,
  GGUF_Q8_0    = 8,
  GGUF_Q8_1    = 9,
  GGUF_Q2_K    = 10,
  GGUF_Q3_K_S  = 11,
  GGUF_Q3_K_M  = 12,
  GGUF_Q3_K_L  = 13,
  GGUF_Q4_K_S  = 14,
  GGUF_Q4_K_M  = 15,
  GGUF_Q5_K_S  = 16,
  GGUF_Q5_K_M  = 17,
  GGUF_Q6_K    = 18
};

typedef struct {
    char *key;
    enum gguf_type type;
    union {
      uint8_t  u8;
      int8_t   i8;
      uint16_t u16;
      int16_t  i16;
      uint32_t u32;
      int32_t  i32;
      float    f32;
      int      b;
      uint64_t u64;
      int64_t  i64;
      double   f64;
      struct {
        char *str;
        size_t len;
      } str;
      struct {
        enum gguf_type type;
        size_t len;
        void *data;
      } arr;
    } value;
} gguf_kv_t;

typedef struct {
  char *name;
  uint32_t n_dims;
  uint64_t *dims;
  enum gguf_tensor_type type;
  uint64_t offset;
} gguf_tensor_info_t;

typedef struct {
  FILE *fp;
  uint32_t version;
  uint64_t n_tensors;
  uint64_t n_kv;
  gguf_kv_t *kvs;
  gguf_tensor_info_t *tensors;
  size_t data_offset;
} gguf_file_t;

static uint64_t read_u64(FILE *fp) {
  uint64_t v;
  fread(&v, 1, 8, fp);
  return v;
}
static uint32_t read_u32(FILE *fp) {
  uint32_t v;
  fread(&v, 1, 4, fp);
  return v;
}
static uint16_t read_u16(FILE *fp) {
  uint16_t v;
  fread(&v, 1, 2, fp);
  return v;
}
static int16_t read_i16(FILE *fp) {
  int16_t v;
  fread(&v, 1, 2, fp);
  return v;
}
static float read_f32(FILE *fp) {
  float v;
  fread(&v, 1, 4, fp);
  return v;
}
static double read_f64(FILE *fp) {
  double v;
  fread(&v, 1, 8, fp);
  return v;
}
static uint8_t read_u8(FILE *fp) {
  uint8_t v;
  fread(&v, 1, 1, fp);
  return v;
}

static char *read_string(FILE *fp) {
  uint64_t len = read_u64(fp);
  char *str = malloc(len + 1);
  if (len > 0) {
    fread(str, 1, len, fp);
  }
  str[len] = 0;
  return str;
}

static void *read_value(FILE *fp, enum gguf_type type) {
  void *p = calloc(1, 64);
  switch (type) {
    case GGUF_TYPE_UINT8: {
      *(uint8_t*)p = read_u8(fp);
      break;
    }
    case GGUF_TYPE_INT8: {
      int8_t v;
      fread(&v,1,1,fp);
      *(int8_t*)p = v;
      break;
    }
    case GGUF_TYPE_UINT16: {
      *(uint16_t*)p = read_u16(fp);
      break;
    }
    case GGUF_TYPE_INT16: {
      *(int16_t*)p = read_i16(fp);
      break;
    }
    case GGUF_TYPE_UINT32: {
      *(uint32_t*)p = read_u32(fp);
      break;
    }
    case GGUF_TYPE_INT32: {
      int32_t v;
      fread(&v,1,4,fp);
      *(int32_t*)p = v;
      break;
    }
    case GGUF_TYPE_FLOAT32: {
      *(float*)p = read_f32(fp);
      break;
    }
    case GGUF_TYPE_BOOL: {
      *(int*)p = read_u8(fp);
      break;
    }
    case GGUF_TYPE_UINT64: {
      *(uint64_t*)p = read_u64(fp);
      break;
    }
    case GGUF_TYPE_INT64: {
      int64_t v;
      fread(&v,1,8,fp);
      *(int64_t*)p = v;
      break;
    }
    case GGUF_TYPE_FLOAT64: {
      *(double*)p = read_f64(fp);
      break;
    }
    case GGUF_TYPE_STRING: {
      char *s = read_string(fp);
      struct {
        char *s;
        size_t l;
      } *sp = p;
      sp->s = s;
      sp->l = strlen(s);
      break;
    }
    case GGUF_TYPE_ARRAY: {
      /* Skip array: read type, length, then skip all elements */
      uint32_t arr_type = read_u32(fp);
      uint64_t arr_len = read_u64(fp);
      for (uint64_t i = 0; i < arr_len; i++) {
        /* Skip each element based on type */
        switch (arr_type) {
          case GGUF_TYPE_UINT8:
          case GGUF_TYPE_INT8:
          case GGUF_TYPE_BOOL: {
            read_u8(fp);
            break;
          }
          case GGUF_TYPE_UINT16:
          case GGUF_TYPE_INT16: {
            read_u16(fp);
            break;
          }
          case GGUF_TYPE_UINT32:
          case GGUF_TYPE_INT32:
          case GGUF_TYPE_FLOAT32: {
            read_u32(fp);
            break;
          }
          case GGUF_TYPE_UINT64:
          case GGUF_TYPE_INT64:
          case GGUF_TYPE_FLOAT64: {
            read_u64(fp);
            break;
          }
          case GGUF_TYPE_STRING: {
            read_string(fp);
            break;
          }
          default: {
            fprintf(
              stderr, "Unknown array element type %d\n", arr_type
            );
            break;
          }
        }
      }
      break;
    }
    default: {
      fprintf(stderr, "Unknown KV type %d\n", type);
      break;
    }
  }
  return p;
}

static gguf_file_t gguf_open(const char *path) {
  gguf_file_t gf = {0};
  gf.fp = fopen(path, "rb");
  if (!gf.fp) {
    fprintf(stderr, "Cannot open %s\n", path);
    exit(1);
  }

  uint32_t magic = read_u32(gf.fp);
  if (magic != GGUF_MAGIC) {
    fprintf(stderr, "Bad magic: 0x%x\n", magic);
    exit(1);
  }
  gf.version = read_u32(gf.fp);
  gf.n_tensors = read_u64(gf.fp);
  gf.n_kv = read_u64(gf.fp);

  printf(
    "GGUF v%d, %llu tensors, %llu metadata\n", gf.version,
    (unsigned long long)gf.n_tensors, (unsigned long long)gf.n_kv
  );

  gf.kvs = calloc(gf.n_kv, sizeof(gguf_kv_t));
  for (uint64_t i = 0; i < gf.n_kv; i++) {
    gf.kvs[i].key = read_string(gf.fp);
    gf.kvs[i].type = read_u32(gf.fp);
    void *vp = read_value(gf.fp, gf.kvs[i].type);
    memcpy(&gf.kvs[i].value, vp, 64);
    free(vp);
  }

  gf.tensors = calloc(gf.n_tensors, sizeof(gguf_tensor_info_t));
  for (uint64_t i = 0; i < gf.n_tensors; i++) {
    gf.tensors[i].name = read_string(gf.fp);
    gf.tensors[i].n_dims = read_u32(gf.fp);
    gf.tensors[i].dims = calloc(gf.tensors[i].n_dims, sizeof(uint64_t));
    for (uint32_t d = 0; d < gf.tensors[i].n_dims; d++) {
      gf.tensors[i].dims[d] = read_u64(gf.fp);
    }
    gf.tensors[i].type = read_u32(gf.fp);
    gf.tensors[i].offset = read_u64(gf.fp);
  }

  long pos = ftell(gf.fp);
  gf.data_offset = (pos + 31) & ~31UL;
  return gf;
}

static size_t gguf_tensor_size(gguf_file_t *gf, int idx) {
  size_t n = 1;
  for (uint32_t d = 0; d < gf->tensors[idx].n_dims; d++) {
    n *= gf->tensors[idx].dims[d];
  }

  enum gguf_tensor_type t = gf->tensors[idx].type;
  switch (t) {
    case GGUF_F32:  return n * 4;
    case GGUF_F16:  return n * 2;
    case GGUF_Q8_0: return (n / 32) * 33;
    case GGUF_Q4_0: return (n / 32) * 18;
    case GGUF_Q4_1: return (n / 32) * 20;
    case GGUF_Q5_0: return (n / 32) * 22;
    case GGUF_Q5_1: return (n / 32) * 24;
    case GGUF_Q8_1: return (n / 32) * 34;
    case GGUF_Q4_K_M: return (n / 256) * 144;
    case GGUF_Q5_K_M: return (n / 256) * 176;
    case GGUF_Q6_K:  return (n / 256) * 210;
    default: return n * 2;
  }
}

static void gguf_read_tensor_data(gguf_file_t *gf, int idx, void *buf) {
  uint64_t off = gf->data_offset + gf->tensors[idx].offset;
  fseek(gf->fp, off, SEEK_SET);
  size_t sz = gguf_tensor_size(gf, idx);
  fread(buf, 1, sz, gf->fp);
}

static int gguf_find_tensor(gguf_file_t *gf, const char *name) {
  for (uint64_t i = 0; i < gf->n_tensors; i++) {
    if (strcmp(gf->tensors[i].name, name) == 0) return (int)i;
  }
  return -1;
}

static const char *gguf_get_kv_str(gguf_file_t *gf, const char *key) {
  for (uint64_t i = 0; i < gf->n_kv; i++) {
    if (
      strcmp(gf->kvs[i].key, key) == 0 &&
      gf->kvs[i].type == GGUF_TYPE_STRING
    ) {
      return gf->kvs[i].value.str.str;
    }
  }
  return NULL;
}

static int64_t gguf_get_kv_i64(gguf_file_t *gf, const char *key, int64_t def) {
  for (uint64_t i = 0; i < gf->n_kv; i++) {
    if (strcmp(gf->kvs[i].key, key) == 0) {
      switch (gf->kvs[i].type) {
        case GGUF_TYPE_INT32: return gf->kvs[i].value.i32;
        case GGUF_TYPE_INT64: return gf->kvs[i].value.i64;
        case GGUF_TYPE_UINT32: return (int64_t)gf->kvs[i].value.u32;
        case GGUF_TYPE_UINT64: return (int64_t)gf->kvs[i].value.u64;
        default: break;
      }
    }
  }
  return def;
}

/* ============================================================
 * Dequantization
 * ============================================================ */

static void dequant_q4_0(float *out, const uint8_t *raw, int n) {
  int n_blocks = n / 32;
  for (int b = 0; b < n_blocks; b++) {
    float d = *(const float*)(raw + b * 18 + 16);
    const uint8_t *qs = raw + b * 18;
    for (int i = 0; i < 16; i++) {
      uint8_t q = qs[i];
      out[b * 32 + i * 2 + 0] = (float)((q >> 0) & 0xF) * d;
      out[b * 32 + i * 2 + 1] = (float)((q >> 4) & 0xF) * d;
    }
  }
}

static void dequant_q4_1(float *out, const uint8_t *raw, int n) {
  int n_blocks = n / 32;
  for (int b = 0; b < n_blocks; b++) {
    float d = *(const float*)(raw + b * 20 + 16);
    float m = *(const float*)(raw + b * 20 + 20);
    const uint8_t *qs = raw + b * 20;
    for (int i = 0; i < 16; i++) {
      uint8_t q = qs[i];
      out[b * 32 + i * 2 + 0] = (float)((q >> 0) & 0xF) * d + m;
      out[b * 32 + i * 2 + 1] = (float)((q >> 4) & 0xF) * d + m;
    }
  }
}

static void dequant_q8_0(float *out, const uint8_t *raw, int n) {
  int n_blocks = n / 32;
  for (int b = 0; b < n_blocks; b++) {
    float d = *(const float*)(raw + b * 33 + 32);
    const int8_t *qs = (const int8_t*)(raw + b * 33);
    for (int i = 0; i < 32; i++) {
      out[b * 32 + i] = (float)qs[i] * d;
    }
  }
}

static void convert_f16_to_f32(float *out, const void *raw, int n) {
  const uint16_t *in = raw;
  for (int i = 0; i < n; i++) {
    uint16_t h = in[i];
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
      f = sign | (mantissa << 13);
    } else if (exp == 31) {
      f = sign | 0x7F800000 | (mantissa << 13);
    } else {
      f = sign | ((exp + 127 - 15) << 23) | (mantissa << 13);
    }
    memcpy(out + i, &f, 4);
  }
}

static float *dequant_tensor(gguf_file_t *gf, int idx) {
  size_t ne = 1;
  for (uint32_t d = 0; d < gf->tensors[idx].n_dims; d++) {
    ne *= gf->tensors[idx].dims[d];
  }

  size_t raw_sz = gguf_tensor_size(gf, idx);
  uint8_t *raw = malloc(raw_sz);
  gguf_read_tensor_data(gf, idx, raw);

  float *out = malloc(ne * sizeof(float));
  enum gguf_tensor_type t = gf->tensors[idx].type;

  /* Debug: print first tensor info */
  static int printed = 0;
  if (!printed) {
    fprintf(
      stderr, "Tensor '%s': type=%d, ne=%zu, raw_sz=%zu\n",
      gf->tensors[idx].name, t, ne, raw_sz
    );
    printed = 1;
  }

  switch (t) {
    case GGUF_F32: {
      memcpy(out, raw, ne * 4);
      break;
    }
    case GGUF_F16: {
      convert_f16_to_f32(out, raw, ne);
      break;
    }
    case GGUF_Q4_0: {
      dequant_q4_0(out, raw, ne);
      break;
    }
    case GGUF_Q4_1: {
      dequant_q4_1(out, raw, ne);
      break;
    }
    case GGUF_Q8_0: {
      dequant_q8_0(out, raw, ne);
      break;
    }
    default: {
      fprintf(
        stderr, "Warning: unsupported quant type %d for tensor %s\n",
        t, gf->tensors[idx].name
      );
      /* Zero out instead ofmemcpy which could read past buffer */
      memset(out, 0, ne * sizeof(float));
      break;
    }
  }

  free(raw);
  return out;
}

/* ============================================================
 * Math Operations
 * ============================================================ */

static void rms_norm(float *out, const float *x, int n, float eps) {
  float sum = 0;
  for (int i = 0; i < n; i++) sum += x[i] * x[i];
  float inv = 1.0f / sqrtf(sum / n + eps);
  for (int i = 0; i < n; i++) out[i] = x[i] * inv;
}

static void matmul_f32(
  float *out, const float *x, const float *w, int M, int N, int K
) {
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      float sum = 0;
      int k = 0;
      #if defined(__aarch64__)
      float32x4_t acc = vdupq_n_f32(0);
      for (; k + 3 < K; k += 4) {
        float32x4_t xv = vld1q_f32(&x[m * K + k]);
        float32x4_t wv = vld1q_f32(&w[n * K + k]);
        acc = vfmaq_f32(acc, xv, wv);
      }
      float32x2_t sum2 = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
      sum = vget_lane_f32(vadd_f32(sum2, vdup_lane_f32(sum2, 1)), 0);
      #elif defined(__ARM_NEON)
      float32x4_t acc = vdupq_n_f32(0);
      for (; k + 3 < K; k += 4) {
        float32x4_t xv = vld1q_f32(&x[m * K + k]);
        float32x4_t wv = vld1q_f32(&w[n * K + k]);
        acc = vaddq_f32(acc, vmulq_f32(xv, wv));
      }
      float32x2_t sum2 = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
      sum = vget_lane_f32(vadd_f32(sum2, vdup_lane_f32(sum2, 1)), 0);
      #endif
      for (; k < K; k++) {
        sum += x[m * K + k] * w[n * K + k];
      }
      out[m * N + n] = sum;
    }
  }
}

static void softmax(float *out, const float *x, int n) {
  float mx = -1e30f;
  for (int i = 0; i < n; i++) {
    if (x[i] > mx) mx = x[i];
  }
  float sum = 0;
  for (int i = 0; i < n; i++) {
    out[i] = expf(x[i] - mx);
    sum += out[i];
  }
  float inv = 1.0f / sum;
  for (int i = 0; i < n; i++) out[i] *= inv;
}

static void silu(float *out, const float *x, int n) {
  for (int i = 0; i < n; i++) {
    out[i] = x[i] / (1.0f + expf(-x[i]));
  }
}

static void apply_rope(float *q, float *k, int dim, int pos, int n_heads) {
  for (int i = 0; i < dim; i += 2) {
    float freq = 1.0f / powf(10000.0f, (float)i / dim);
    float angle = (float)pos * freq;
    float cos_a = cosf(angle), sin_a = sinf(angle);
    for (int h = 0; h < n_heads; h++) {
      int idx = h * dim + i;
      float q0 = q[idx], q1 = q[idx + 1];
      float k0 = k[idx], k1 = k[idx + 1];
      q[idx]     = q0 * cos_a - q1 * sin_a;
      q[idx + 1] = q0 * sin_a + q1 * cos_a;
      k[idx]     = k0 * cos_a - k1 * sin_a;
      k[idx + 1] = k0 * sin_a + k1 * cos_a;
    }
  }
}

static float sample_argmax(const float *logits, int n) {
  int id = 0;
  float mx = -1e30f;
  for (int i = 0; i < n; i++) {
    if (logits[i] > mx) {
      mx = logits[i];
      id = i;
    }
  }
  return (float)id;
}

static float sample_temperature(const float *logits, int n, float temp) {
  float scaled[32000];
  int nc = n < 32000 ? n : 32000;
  for (int i = 0; i < nc; i++) scaled[i] = logits[i] / temp;
  softmax(scaled, scaled, nc);
  float r = (float)rand() / (float)(RAND_MAX + 1.0);
  float cum = 0;
  for (int i = 0; i < nc; i++) {
    cum += scaled[i];
    if (r <= cum) return (float)i;
  }
  return (float)(nc - 1);
}

/* ============================================================
 * Tokenizer
 * ============================================================ */

typedef struct {
  char **tokens;
  int *scores;
  int n_vocab;
} tokenizer_t;

static tokenizer_t load_vocab(gguf_file_t *gf) {
  tokenizer_t tok = {0};

  /* Simple fallback tokenizer - works with any model */
  tok.n_vocab = 32000;
  tok.tokens = calloc(tok.n_vocab, sizeof(char*));
  tok.scores = calloc(tok.n_vocab, sizeof(int));

  /* Try to find actual vocab from GGUF */
  for (uint64_t i = 0; i < gf->n_kv; i++) {
    if (
      strcmp(gf->kvs[i].key, "tokenizer.ggml.tokens") == 0 &&
      gf->kvs[i].type == GGUF_TYPE_ARRAY &&
      gf->kvs[i].value.arr.len > 0
    ) {
      tok.n_vocab = gf->kvs[i].value.arr.len;
      printf("Found vocab: %d tokens\n", tok.n_vocab);
      break;
    }
  }

  /* Generate placeholder tokens */
  for (int i = 0; i < tok.n_vocab; i++) {
    tok.tokens[i] = malloc(8);
    snprintf(tok.tokens[i], 8, "<%d>", i);
  }

  printf("Vocab: %d tokens\n", tok.n_vocab);
  return tok;
}

static int tokenize(
  tokenizer_t *tok, const char *text, int *ids, int max_ids
) {
  int n = 0;
  const char *p = text;
  while (*p && n < max_ids) {
    int best_len = 1;
    int best_id = 0;
    for (int i = 0; i < tok->n_vocab; i++) {
      int len = strlen(tok->tokens[i]);
      if (len <= (int)strlen(p) && strncmp(p, tok->tokens[i], len) == 0) {
        if (len > best_len) {
          best_len = len;
          best_id = i;
        }
      }
    }
    ids[n++] = best_id;
    p += best_len;
  }
  return n;
}

static const char *detokenize(tokenizer_t *tok, int id) {
  if (id >= 0 && id < tok->n_vocab) return tok->tokens[id];
  return "?";
}

/* ============================================================
 * Model
 * ============================================================ */

typedef struct {
  int n_layers, n_heads, n_embd, n_ff, n_vocab, ctx_len;
  int head_dim, n_kv_heads;

  float *token_embd;
  float *output_norm_w;
  float *output_proj_w;

  float **attn_norm_w;
  float **q_proj_w;
  float **k_proj_w;
  float **v_proj_w;
  float **o_proj_w;
  float **ffn_norm_w;
  float **gate_proj_w;
  float **up_proj_w;
  float **down_proj_w;
} model_t;

static int find_tensor(gguf_file_t *gf, const char *fmt, int layer) {
  char name[128];
  snprintf(name, sizeof(name), fmt, layer);
  return gguf_find_tensor(gf, name);
}

static model_t load_model(gguf_file_t *gf) {
  model_t m = {0};
  const char *arch = gguf_get_kv_str(gf, "general.architecture");
  if (arch) printf("Architecture: %s\n", arch);

  m.n_layers = (int)gguf_get_kv_i64(
    gf, "llama.attention.layer_count",
    gguf_get_kv_i64(gf, "block_count", 32)
  );
  m.n_heads = (int)gguf_get_kv_i64(
    gf, "llama.attention.head_count",
    gguf_get_kv_i64(gf, "attention.head_count", 32)
  );
  m.n_embd = (int)gguf_get_kv_i64(
    gf, "llama.embedding_length",
    gguf_get_kv_i64(gf, "embedding_length", 4096)
  );
  m.n_ff = (int)gguf_get_kv_i64(
    gf, "llama.feed_forward_length",
    gguf_get_kv_i64(gf, "feed_forward_length", 11008)
  );
  m.ctx_len = (int)gguf_get_kv_i64(
    gf, "llama.context_length",
    gguf_get_kv_i64(gf, "context_length", 2048)
  );
  m.n_kv_heads = (int)gguf_get_kv_i64(
    gf, "llama.attention.head_count_kv",
    gguf_get_kv_i64(gf, "attention.head_count_kv", m.n_heads)
  );
  m.head_dim = m.n_embd / m.n_heads;

  printf(
    "Model: %d layers, %d heads, %d embd, %d kv_heads\n",
    m.n_layers, m.n_heads, m.n_embd, m.n_kv_heads
  );

  m.attn_norm_w = calloc(m.n_layers, sizeof(float*));
  m.q_proj_w = calloc(m.n_layers, sizeof(float*));
  m.k_proj_w = calloc(m.n_layers, sizeof(float*));
  m.v_proj_w = calloc(m.n_layers, sizeof(float*));
  m.o_proj_w = calloc(m.n_layers, sizeof(float*));
  m.ffn_norm_w = calloc(m.n_layers, sizeof(float*));
  m.gate_proj_w = calloc(m.n_layers, sizeof(float*));
  m.up_proj_w = calloc(m.n_layers, sizeof(float*));
  m.down_proj_w = calloc(m.n_layers, sizeof(float*));

  for (int l = 0; l < m.n_layers; l++) {
    int idx;

    idx = find_tensor(gf, "blk.%d.attn_norm.weight", l);
    if (idx < 0) idx = find_tensor(gf, "layer.%d.attention_norm.weight", l);
    if (idx >= 0) m.attn_norm_w[l] = dequant_tensor(gf, idx);

    idx = find_tensor(gf, "blk.%d.attn_q.weight", l);
    if (idx < 0) idx = find_tensor(gf, "layer.%d.attention.wq.weight", l);
    if (idx >= 0) m.q_proj_w[l] = dequant_tensor(gf, idx);

    idx = find_tensor(gf, "blk.%d.attn_k.weight", l);
    if (idx < 0) idx = find_tensor(gf, "layer.%d.attention.wk.weight", l);
    if (idx >= 0) m.k_proj_w[l] = dequant_tensor(gf, idx);

    idx = find_tensor(gf, "blk.%d.attn_v.weight", l);
    if (idx < 0) idx = find_tensor(gf, "layer.%d.attention.wv.weight", l);
    if (idx >= 0) m.v_proj_w[l] = dequant_tensor(gf, idx);

    idx = find_tensor(gf, "blk.%d.attn_output.weight", l);
    if (idx < 0) idx = find_tensor(gf, "layer.%d.attention.wo.weight", l);
    if (idx >= 0) m.o_proj_w[l] = dequant_tensor(gf, idx);

    idx = find_tensor(gf, "blk.%d.ffn_norm.weight", l);
    if (idx < 0) idx = find_tensor(gf, "layer.%d.ffn_norm.weight", l);
    if (idx >= 0) m.ffn_norm_w[l] = dequant_tensor(gf, idx);

    idx = find_tensor(gf, "blk.%d.ffn_gate.weight", l);
    if (idx < 0) idx = find_tensor(gf, "layer.%d.feed_forward.w3.weight", l);
    if (idx >= 0) m.gate_proj_w[l] = dequant_tensor(gf, idx);

    idx = find_tensor(gf, "blk.%d.ffn_up.weight", l);
    if (idx < 0) idx = find_tensor(gf, "layer.%d.feed_forward.w1.weight", l);
    if (idx >= 0) m.up_proj_w[l] = dequant_tensor(gf, idx);

    idx = find_tensor(gf, "blk.%d.ffn_down.weight", l);
    if (idx < 0) idx = find_tensor(gf, "layer.%d.feed_forward.w2.weight", l);
    if (idx >= 0) m.down_proj_w[l] = dequant_tensor(gf, idx);

    printf("\r  Loading layer %d/%d...", l + 1, m.n_layers);
  }
  printf("\n");

  int idx = gguf_find_tensor(gf, "token_embd.weight");
  if (idx < 0) idx = gguf_find_tensor(gf, "embed_tokens.weight");
  if (idx >= 0) {
    m.n_vocab = gf->tensors[idx].dims[0];
    m.token_embd = dequant_tensor(gf, idx);
  }

  idx = gguf_find_tensor(gf, "output_norm.weight");
  if (idx < 0) idx = gguf_find_tensor(gf, "norm.weight");
  if (idx >= 0) m.output_norm_w = dequant_tensor(gf, idx);

  idx = gguf_find_tensor(gf, "output.weight");
  if (idx >= 0) m.output_proj_w = dequant_tensor(gf, idx);

  return m;
}

/* ============================================================
 * Inference
 * ============================================================ */

static int generate(
  model_t *m, tokenizer_t *tok, const char *prompt,
  char *out, int max_out, int max_tokens, float temp
) {
  int tokens[4096];
  int n_tokens = tokenize(tok, prompt, tokens, 4096);

  int kv_dim = m->n_embd;
  int kv_len = m->ctx_len;

  float *hidden = calloc(m->n_embd, sizeof(float));
  float *residual = calloc(m->n_embd, sizeof(float));
  float *q = calloc(m->n_embd, sizeof(float));
  float *k = calloc(m->n_embd, sizeof(float));
  float *v = calloc(m->n_embd, sizeof(float));
  float *att_out = calloc(m->n_embd, sizeof(float));
  float *ff_tmp = calloc(m->n_ff * 2, sizeof(float));
  float *att_scores = calloc(m->n_heads * kv_len, sizeof(float));

  float *k_cache = calloc(m->n_layers * kv_len * kv_dim, sizeof(float));
  float *v_cache = calloc(m->n_layers * kv_len * kv_dim, sizeof(float));

  int group_size = m->n_heads / m->n_kv_heads;

  out[0] = 0;
  int out_pos = 0;

  srand(time(NULL));

  for (int t = 0; t < n_tokens + max_tokens; t++) {
    int tok_id;
    if (t < n_tokens) {
      tok_id = tokens[t];
    } else {
      float tid;
      if (temp > 0.01f) {
        tid = sample_temperature(hidden, m->n_vocab, temp);
      } else {
        tid = sample_argmax(hidden, m->n_vocab);
      }
      tok_id = (int)tid;

      const char *s = detokenize(tok, tok_id);
      int slen = strlen(s);
      if (out_pos + slen < max_out - 1) {
        memcpy(out + out_pos, s, slen);
        out_pos += slen;
        out[out_pos] = 0;
      }
      if (t - n_tokens >= max_tokens) break;
    }

    for (int i = 0; i < m->n_embd; i++) {
      hidden[i] = m->token_embd[tok_id * m->n_embd + i];
    }

    for (int l = 0; l < m->n_layers; l++) {
      rms_norm(residual, hidden, m->n_embd, 1e-5f);

      /* QKV - skip if weights missing */
      if (m->q_proj_w[l] && m->k_proj_w[l] && m->v_proj_w[l]) {
        matmul_f32(
          q, residual, m->q_proj_w[l], 1, m->n_embd, m->n_embd
        );
        matmul_f32(
          k, residual, m->k_proj_w[l], 1,
          m->n_kv_heads * m->head_dim, m->n_embd
        );
        matmul_f32(
          v, residual, m->v_proj_w[l], 1,
          m->n_kv_heads * m->head_dim, m->n_embd
        );
      } else {
        /* Skip this layer if weights missing */
        continue;
      }

      apply_rope(q, k, m->head_dim, t, m->n_heads);

      int co = l * kv_len * kv_dim + (t % kv_len) * kv_dim;
      memcpy(k_cache + co, k, kv_dim * sizeof(float));
      memcpy(v_cache + co, v, kv_dim * sizeof(float));

      float scale = 1.0f / sqrtf(m->head_dim);
      for (int h = 0; h < m->n_heads; h++) {
        float *qh = q + h * m->head_dim;
        int kv_h = h / group_size;

        for (int t2 = 0; t2 <= t && t2 < kv_len; t2++) {
          float *kh = k_cache + l * kv_len * kv_dim + t2 * kv_dim + kv_h * m->head_dim;
          float dot = 0;
          for (int d = 0; d < m->head_dim; d++) {
            dot += qh[d] * kh[d];
          }
          att_scores[h * kv_len + t2] = dot * scale;
        }

        int len = t + 1;
        if (len > kv_len) len = kv_len;
        softmax(att_scores + h * kv_len, att_scores + h * kv_len, len);

        float *oh = att_out + h * m->head_dim;
        memset(oh, 0, m->head_dim * sizeof(float));
        for (int t2 = 0; t2 < len; t2++) {
          float a = att_scores[h * kv_len + t2];
          float *vh = v_cache + l * kv_len * kv_dim + t2 * kv_dim + kv_h * m->head_dim;
          for (int d = 0; d < m->head_dim; d++) {
            oh[d] += a * vh[d];
          }
        }
      }

      if (m->o_proj_w[l]) {
        matmul_f32(
          residual, att_out, m->o_proj_w[l], 1, m->n_embd, m->n_embd
        );
        for (int i = 0; i < m->n_embd; i++) {
          hidden[i] += residual[i];
        }
      } else {
        for (int i = 0; i < m->n_embd; i++) {
          hidden[i] += att_out[i];
        }
      }

      rms_norm(residual, hidden, m->n_embd, 1e-5f);

      if (m->gate_proj_w[l] && m->up_proj_w[l]) {
        float *gate = ff_tmp;
        float *up = ff_tmp + m->n_ff;
        matmul_f32(
          gate, residual, m->gate_proj_w[l], 1, m->n_ff, m->n_embd
        );
        matmul_f32(
          up, residual, m->up_proj_w[l], 1, m->n_ff, m->n_embd
        );
        silu(gate, gate, m->n_ff);
        for (int i = 0; i < m->n_ff; i++) {
          gate[i] *= up[i];
        }
        matmul_f32(
          residual, gate, m->down_proj_w[l], 1, m->n_embd, m->n_ff
        );
        for (int i = 0; i < m->n_embd; i++) {
          hidden[i] += residual[i];
        }
      }
    }

    rms_norm(hidden, hidden, m->n_embd, 1e-5f);

    if (m->output_proj_w) {
      matmul_f32(residual, hidden, m->output_proj_w, 1, m->n_vocab, m->n_embd);
      memcpy(hidden, residual, m->n_vocab * sizeof(float));
    }
  }

  free(hidden); free(residual); free(q); free(k); free(v);
  free(att_out); free(ff_tmp); free(att_scores);
  free(k_cache); free(v_cache);
  return out_pos;
}

/* ============================================================
 * HTTP Server
 * ============================================================ */

static char *read_http_body(int fd) {
  static char buf[131072];
  int total = 0;
  while (total < (int)sizeof(buf) - 1) {
    int n = recv(fd, buf + total, sizeof(buf) - total - 1, 0);
    if (n <= 0) break;
    total += n;
    buf[total] = 0;
    if (strstr(buf, "\r\n\r\n")) break;
  }
  char *body = strstr(buf, "\r\n\r\n");
  return strdup(body ? body + 4 : buf);
}

static void http_response(int fd, int code, const char *json) {
  char hdr[512];
  snprintf(
    hdr, sizeof(hdr),
    "HTTP/1.1 %d %s\r\n"
    "Content-Type: application/json\r\n"
    "Content-Length: %zu\r\n"
    "Access-Control-Allow-Origin: *\r\n"
    "Connection: close\r\n\r\n",
    code, code == 200 ? "OK" : "Error",
    strlen(json)
  );
  send(fd, hdr, strlen(hdr), 0);
  send(fd, json, strlen(json), 0);
}

static int json_str(const char *json, const char *key, char *val, int maxlen) {
  char needle[64];
  snprintf(needle, sizeof(needle), "\"%s\"", key);
  const char *p = strstr(json, needle);
  if (!p) return 0;
  p = strchr(p + strlen(needle), ':');
  if (!p) return 0;
  p++;
  while (*p == ' ') p++;
  if (*p == '"') {
    p++;
    int i = 0;
    while (*p && *p != '"' && i < maxlen - 1) val[i++] = *p++;
    val[i] = 0;
    return 1;
  }
  return 0;
}

static int json_int(const char *json, const char *key, int def) {
  char val[64];
  if (json_str(json, key, val, sizeof(val))) return atoi(val);
  return def;
}

static float json_float(const char *json, const char *key, float def) {
  char val[64];
  if (json_str(json, key, val, sizeof(val))) return (float)atof(val);
  return def;
}

static model_t *g_model;
static tokenizer_t *g_tok;

static void handle_request(int fd) {
  char *body = read_http_body(fd);

  if (strstr(body, "OPTIONS")) {
    http_response(fd, 200, "{}");
    free(body);
    return;
  }

  if (strstr(body, "prompt")) {
    char prompt[8192] = {0};
    int max_tokens = json_int(body, "max_tokens", 256);
    float temp = json_float(body, "temperature", 0.8f);

    if (
      json_str(body, "prompt", prompt, sizeof(prompt)) ||
      json_str(body, "content", prompt, sizeof(prompt))
    ) {

      char *output = calloc(65536, 1);
      generate(g_model, g_tok, prompt, output, 65536, max_tokens, temp);

      char *escaped = calloc(strlen(output) * 2 + 1, 1);
      int j = 0;
      for (int i = 0; output[i]; i++) {
        if (output[i] == '"') { escaped[j++] = '\\'; escaped[j++] = '"'; }
        else if (output[i] == '\\') { escaped[j++] = '\\'; escaped[j++] = '\\'; }
        else if (output[i] == '\n') { escaped[j++] = '\\'; escaped[j++] = 'n'; }
        else escaped[j++] = output[i];
      }

      char json[80000];
      snprintf(
        json, sizeof(json),
        "{\"response\":\"%s\",\"model\":\"mini-llm\",\"done\":true}",
        escaped
      );
      http_response(fd, 200, json);
      free(output);
      free(escaped);
    } else {
        http_response(fd, 400, "{\"error\":\"missing prompt\"}");
    }
  }
  else if (strstr(body, "GET /api/tags") || strstr(body, "/api/tags")) {
    http_response(
      fd, 200,
      "{\"models\":[{\"name\":\"mini-llm\",\"size\":0,\"parameter_size\":\"?\"}]}"
    );
  }
  else if (strstr(body, "/api/version")) {
    http_response(fd, 200, "{\"version\":\"0.1.0\"}");
  }
  else {
    http_response(fd, 404, "{\"error\":\"not found\"}");
  }
  free(body);
}

static void *server_thread(void *arg) {
  (void)arg;
  int server_fd = socket(AF_INET, SOCK_STREAM, 0);
  int opt = 1;
  setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  struct sockaddr_in addr = {
    .sin_family = AF_INET,
    .sin_port = htons(8080),
    .sin_addr.s_addr = INADDR_ANY
  };

  if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
    perror("bind");
    exit(1);
  }
  listen(server_fd, 16);
  printf("\nServer running at http://localhost:8080\n");
  printf("  curl http://localhost:8080/api/generate -d '{\"prompt\":\"Hello\"}'\n\n");

  while (1) {
    int client_fd = accept(server_fd, NULL, NULL);
    if (client_fd >= 0) {
      handle_request(client_fd);
      close(client_fd);
    }
  }
  return NULL;
}

/* ============================================================
 * Main
 * ============================================================ */

int main(int argc, char **argv) {
  printf("mini-llm v0.1.0 - Single file LLM inference\n");
  printf("Build: %s %s\n", __DATE__, __TIME__);
  #if HAS_NEON
  printf("SIMD: ARM NEON enabled\n");
  #else
  printf("SIMD: none (scalar)\n");
  #endif

  if (argc < 2) {
    printf("\nUsage: %s model.gguf [prompt]\n", argv[0]);
    printf("  API: curl http://localhost:8080/api/generate -d '{\"prompt\":\"Hello\"}'\n");
    return 1;
  }

  printf("\nLoading: %s\n", argv[1]);
  gguf_file_t gf = gguf_open(argv[1]);

  model_t model = load_model(&gf);
  tokenizer_t tok = load_vocab(&gf);
  g_model = &model;
  g_tok = &tok;

  if (argc > 2) {
    char *output = calloc(65536, 1);
    printf("\nPrompt: %s\n", argv[2]);
    printf("Generating...\n\n");
    generate(g_model, g_tok, argv[2], output, 65536, 128, 0.8f);
    printf("%s\n", output);
    free(output);
  } else {
    pthread_t tid;
    pthread_create(&tid, NULL, server_thread, NULL);
    pthread_join(tid, NULL);
  }

  return 0;
}
