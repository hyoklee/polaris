/*
 * NVSHMEM uniqueid-based init test
 * Exchanges the unique ID via a shared file on Lustre (/lus/grand),
 * so no MPI (and no SSH) is needed for bootstrap.
 * Launch with PALS:  mpiexec -n 8 --ppn 4 ./nvshmem_uniqueid_test
 * PALS sets PMI_RANK / PMI_SIZE / PMI_LOCAL_RANK automatically.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "nvshmem.h"
#include "nvshmemx.h"

#define CUDA_CHECK(stmt)                                                          \
    do {                                                                          \
        cudaError_t result = (stmt);                                              \
        if (cudaSuccess != result) {                                              \
            fprintf(stderr, "[%s:%d] cuda failed: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(result));                                  \
            exit(1);                                                              \
        }                                                                         \
    } while (0)

/* Each PE puts its rank value into the *next* PE's target slot. */
__global__ void simple_shift(int *target, int mype, int npes) {
    int peer = (mype + 1) % npes;
    nvshmem_int_p(target, mype, peer);
}

int main(int argc, char *argv[]) {
    /* ── Rank / size from PALS/PMI environment ─────────────────────────── */
    const char *rank_env  = getenv("PMI_RANK");
    const char *size_env  = getenv("PMI_SIZE");
    const char *lrank_env = getenv("PMI_LOCAL_RANK");

    int myrank     = rank_env  ? atoi(rank_env)  : 0;
    int nranks     = size_env  ? atoi(size_env)  : 1;
    int local_rank = lrank_env ? atoi(lrank_env) : 0;

    char hostname[256] = "?";
    gethostname(hostname, sizeof(hostname));
    printf("[rank %d/%d local %d] starting on %s  (GPU %d)\n",
           myrank, nranks, local_rank, hostname, local_rank);
    fflush(stdout);

    CUDA_CHECK(cudaSetDevice(local_rank));

    /* ── Exchange uniqueid via shared Lustre file ──────────────────────── */
    const char *uid_file_env = getenv("NVSHMEM_UNIQUEID_FILE");
    const char *uid_file     = uid_file_env ? uid_file_env
                                            : "/tmp/nvshmem_uid.bin";

    nvshmemx_uniqueid_t uid;
    memset(&uid, 0, sizeof(uid));

    if (myrank == 0) {
        unlink(uid_file);   /* remove stale file */

        if (nvshmemx_get_uniqueid(&uid) != 0) {
            fprintf(stderr, "rank 0: nvshmemx_get_uniqueid failed\n");
            exit(1);
        }

        /* Write to a tmp file then rename for atomic visibility */
        char tmp[512];
        snprintf(tmp, sizeof(tmp), "%s.tmp", uid_file);
        FILE *f = fopen(tmp, "wb");
        if (!f) { perror("fopen tmp"); exit(1); }
        if (fwrite(&uid, sizeof(uid), 1, f) != 1) { perror("fwrite"); exit(1); }
        fclose(f);
        if (rename(tmp, uid_file) != 0) { perror("rename"); exit(1); }
        printf("[rank 0] uniqueid written to %s\n", uid_file);
        fflush(stdout);
    } else {
        /* Poll until rank 0 has written the file (up to 30 s) */
        int waited = 0;
        while (access(uid_file, R_OK) != 0) {
            usleep(200000);   /* 200 ms */
            if (++waited > 150) {
                fprintf(stderr, "rank %d: timeout waiting for %s\n",
                        myrank, uid_file);
                exit(1);
            }
        }
        usleep(200000);   /* extra settling time for Lustre */

        FILE *f = fopen(uid_file, "rb");
        if (!f) { perror("fopen uid"); exit(1); }
        if (fread(&uid, sizeof(uid), 1, f) != 1) { perror("fread"); exit(1); }
        fclose(f);
        printf("[rank %d] uniqueid read from %s\n", myrank, uid_file);
        fflush(stdout);
    }

    /* ── Init NVSHMEM with uniqueid ────────────────────────────────────── */
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    if (nvshmemx_set_attr_uniqueid_args(myrank, nranks, &uid, &attr) != 0) {
        fprintf(stderr, "rank %d: nvshmemx_set_attr_uniqueid_args failed\n",
                myrank);
        exit(1);
    }
    if (nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr) != 0) {
        fprintf(stderr, "rank %d: nvshmemx_init_attr failed\n", myrank);
        exit(1);
    }

    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    printf("[rank %d] NVSHMEM ready: mype=%d  npes=%d\n", myrank, mype, npes);
    fflush(stdout);

    /* ── Symmetric allocation and simple-shift kernel ──────────────────── */
    int *target = (int *)nvshmem_malloc(sizeof(int));
    if (!target) {
        fprintf(stderr, "PE %d: nvshmem_malloc failed\n", mype);
        exit(1);
    }
    CUDA_CHECK(cudaMemset(target, -1, sizeof(int)));
    nvshmem_barrier_all();

    simple_shift<<<1, 1>>>(target, mype, npes);
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    /* Verify: each PE should have received the value from the *previous* PE */
    int expected = (mype - 1 + npes) % npes;
    int received = -1;
    CUDA_CHECK(cudaMemcpy(&received, target, sizeof(int), cudaMemcpyDeviceToHost));

    if (received == expected)
        printf("[PE %d of %d] PASS: got %d from PE %d\n",
               mype, npes, received, expected);
    else
        printf("[PE %d of %d] FAIL: expected %d, got %d\n",
               mype, npes, expected, received);
    fflush(stdout);

    nvshmem_free(target);
    nvshmem_finalize();

    if (myrank == 0) unlink(uid_file);
    return 0;
}
