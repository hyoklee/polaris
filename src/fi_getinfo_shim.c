/*
 * fi_getinfo_shim.c  v4 — diagnostic + CXI probe
 *
 * LD_PRELOAD shim for Polaris Cray libfabric 2.2.0rc1.
 * Logs what NVSHMEM asks for and probes multiple MR-mode combos
 * to find one the CXI 2.2 provider actually accepts.
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>

typedef int (*fi_getinfo_fn_t)(uint32_t, const char *, const char *,
                               uint64_t, const struct fi_info *,
                               struct fi_info **);
static fi_getinfo_fn_t real_fi_getinfo = NULL;
static int shim_done = 0;  /* only run probe once (rank 0) */

static void get_real(void) {
    if (!real_fi_getinfo) {
        real_fi_getinfo = (fi_getinfo_fn_t)dlsym(RTLD_NEXT, "fi_getinfo");
        if (!real_fi_getinfo)
            fprintf(stderr, "[fi_shim] ERROR: dlsym failed\n");
    }
}

static void print_info_list(const char *tag, struct fi_info *list) {
    int n = 0;
    for (struct fi_info *p = list; p; p = p->next, n++) {
        fprintf(stderr, "[fi_shim]   [%d] prov=%s ep=%d mr_mode=0x%lx "
                "inject=%zu nic=%s caps=0x%lx\n",
                n,
                p->fabric_attr ? (p->fabric_attr->prov_name ?: "?") : "?",
                p->ep_attr ? (int)p->ep_attr->type : -1,
                p->domain_attr ? (unsigned long)p->domain_attr->mr_mode : 0UL,
                p->tx_attr ? p->tx_attr->inject_size : 0,
                p->nic ? "yes" : "no",
                (unsigned long)p->caps);
    }
    fprintf(stderr, "[fi_shim] %s: %d result(s)\n", tag, n);
    fflush(stderr);
}

static int try_hints(uint32_t ver, const char *node, const char *svc,
                     uint64_t flags, const char *label,
                     uint64_t caps, uint64_t mode,
                     enum fi_ep_type ep_type, uint64_t mr_mode,
                     size_t inject_sz,
                     struct fi_info **out)
{
    struct fi_info *h = fi_allocinfo();
    if (!h) return -ENOMEM;
    h->caps = caps;
    h->mode = mode;
    h->ep_attr->type = ep_type;
    h->tx_attr->inject_size = inject_sz;
    h->domain_attr->mr_mode = (int)mr_mode;
    int ret = real_fi_getinfo(ver, node, svc, flags, h, out);
    fi_freeinfo(h);
    if (ret == 0) {
        fprintf(stderr, "[fi_shim] PROBE OK  [%s]: mr=0x%lx\n", label, mr_mode);
        print_info_list(label, *out);
    } else {
        fprintf(stderr, "[fi_shim] PROBE FAIL[%s]: mr=0x%lx → %d\n",
                label, mr_mode, ret);
    }
    fflush(stderr);
    return ret;
}

int fi_getinfo(uint32_t version,
               const char *node, const char *service,
               uint64_t flags,
               const struct fi_info *hints,
               struct fi_info **info)
{
    get_real();
    if (!real_fi_getinfo) return -ENOSYS;

    uint32_t cur = fi_version();

    /* Log what NVSHMEM is asking for */
    if (hints && !shim_done) {
        fprintf(stderr, "[fi_shim] NVSHMEM hints: caps=0x%lx mode=0x%lx "
                "mr_mode=0x%lx ep_type=%d inject=%zu prov=%s\n",
                (unsigned long)hints->caps,
                (unsigned long)hints->mode,
                hints->domain_attr ? (unsigned long)hints->domain_attr->mr_mode : 0UL,
                hints->ep_attr ? (int)hints->ep_attr->type : -1,
                hints->tx_attr ? hints->tx_attr->inject_size : 0,
                hints->fabric_attr && hints->fabric_attr->prov_name
                    ? hints->fabric_attr->prov_name : "none");
        fflush(stderr);
    }

    /* Attempt 1: upgrade version, pass original hints */
    int ret = real_fi_getinfo(cur, node, service, flags, hints, info);
    if (ret == 0) {
        fprintf(stderr, "[fi_shim] OK with original hints\n");
        fflush(stderr);
        shim_done = 1;
        return 0;
    }

    if (!shim_done) {
        shim_done = 1;
        uint64_t base_caps = FI_RMA | FI_ATOMIC | FI_HMEM;
        /* Probe different MR mode combos that CXI 2.x might accept */
        struct { const char *name; uint64_t mr; } modes[] = {
            {"LOCAL|VIRT|ALLOC|HMEM",
             FI_MR_LOCAL|FI_MR_VIRT_ADDR|FI_MR_ALLOCATED|FI_MR_HMEM},
            {"LOCAL|VIRT|ALLOC|HMEM|RMA_EVENT",
             FI_MR_LOCAL|FI_MR_VIRT_ADDR|FI_MR_ALLOCATED|FI_MR_HMEM|(1<<6)},
            {"LOCAL|ENDPOINT|HMEM",
             FI_MR_LOCAL|(1<<8)|FI_MR_HMEM},
            {"LOCAL|VIRT|ALLOC",
             FI_MR_LOCAL|FI_MR_VIRT_ADDR|FI_MR_ALLOCATED},
            {"PROV_KEY|VIRT|ALLOC|HMEM",
             FI_MR_PROV_KEY|FI_MR_VIRT_ADDR|FI_MR_ALLOCATED|FI_MR_HMEM},
            {"0 (no mode)", 0},
        };
        for (int i = 0; i < (int)(sizeof(modes)/sizeof(modes[0])); i++) {
            struct fi_info *tmp = NULL;
            if (try_hints(cur, node, service, flags, modes[i].name,
                          base_caps, 0, FI_EP_RDM, modes[i].mr, 8, &tmp) == 0) {
                *info = tmp;
                return 0;
            }
        }
    }

    /* Fallback: NULL hints */
    ret = real_fi_getinfo(cur, node, service, flags, NULL, info);
    fprintf(stderr, "[fi_shim] fallback NULL hints: %s\n", ret ? "FAIL" : "OK");
    fflush(stderr);
    return ret;
}
