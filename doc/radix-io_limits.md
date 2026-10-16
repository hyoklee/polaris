# radix-io Lustre Quota and capacity Queue Limits on Polaris — 2026-10-16

## Summary

The `radix-io` project has a **4 TB** Lustre quota (hard limit **4.4 TB**),
shared by every project member. On 2026-10-16, **1.77 TB** was used, leaving
about **2.2 TB** free.

| Item | Value |
|---|---|
| Soft quota | 4 TB |
| Hard limit (writes fail) | 4.4 TB |
| Used (2026-10-16) | 1.77 TB |
| Free below soft quota | ~2.2 TB |
| File-count limit | none (~15.6 M files in use) |
| Grace period | not running |
| Per-user limit | none |
| hyoklee's own usage | 292.9 GB, 75,167 files |

---

## One quota, two paths

`/eagle/radix-io` and `/grand/radix-io` share the same 4 TB.
`myprojectquotas` lists `/lus/grand` and `/lus/eagle` as separate rows with
identical numbers, but both directories sit on the same Lustre filesystem
(same device id in `stat`) and are charged to the same project quota.
Do not count the space twice.

```
/eagle -> /lus/eagle/projects    /eagle/radix-io = /lus/eagle/projects/radix-io
/grand -> /lus/grand/projects    /grand/radix-io = /lus/grand/projects/radix-io
```

## Soft vs. hard limit

- **4 TB soft quota:** going over starts a grace period.
- **4.4 TB hard limit:** writes fail at this point, regardless of grace.

The quota is project-wide, not per user, so another member's data can fill
it. Of the 1.77 TB used, 292.9 GB belongs to hyoklee; the rest belongs to
other radix-io members.

---

## Checking it

```bash
# Project and personal quotas for every project you belong to
myprojectquotas

# Project quota directly from Lustre (radix-io project id is 2166)
lfs quota -p 2166 -h /lus/eagle

# Your own usage on the filesystem
lfs quota -u $USER -h /lus/eagle
```

Output on 2026-10-16:

```
Name          Type     Filesystem   Used    Quota   Grace
radix-io      Project  /lus/grand   1.77T   4T      -
radix-io      Project  /lus/eagle   1.77T   4T      -

Filesystem    used   quota  limit  grace  files     quota  limit  grace
/lus/eagle    1.77T  4T     4.4T   -      15586392  0      0      -   (project 2166)
/lus/eagle    292.9G 0k     0k     -      75167     0      0      -   (uid 36126)
```

---

## capacity queue limits

`capacity` is the queue for small, long jobs. Of the general queues checked,
it is the one that takes a 1-node job longer than 1 hour without preemption
(`debug` and `debug-scaling` cap at 1 hour; `prod`/`small` need at least
10 nodes; `preemptable` jobs can be killed early).

| Limit | Value |
|---|---|
| Walltime per job | 5 min – **168 hours (7 days)** |
| Nodes per job | 1–4 |
| Running jobs | 1 per project (shared by all of radix-io) |
| Queued jobs | 2 per project (shared by all of radix-io) |
| Total nodes for the queue | 32 (all users) |

The running and queued limits count every radix-io member's jobs, so a
teammate's job in `capacity` makes yours wait. The 3-hourly clio-core GPU
test (`~/bin/clio_core_gpu_dev_c.pbs`, 1 node, 2 hours) runs in this queue.

### Checking it

```bash
qstat -Qf capacity | grep -E 'resources_(max|min)\.(walltime|nodect)|max_run|max_queued'
```

Output on 2026-10-16:

```
    max_queued = [p:PBS_GENERIC=2]
    resources_max.nodect = 4
    resources_max.walltime = 168:00:00
    resources_min.nodect = 1
    resources_min.walltime = 00:05:00
    max_run = [p:PBS_GENERIC=1]
    max_run_res.nodect = [o:PBS_ALL=32]
```
