#!/usr/bin/env bash
# Can this container mount storage by itself, without recreating the Workshop?
#
#   bash scripts/alaya/check_mount_capability.sh
#
# Being root inside a container is not enough. The mount syscall needs
# CAP_SYS_ADMIN, which Kubernetes drops for unprivileged pods, and FUSE-based
# alternatives need /dev/fuse. This reports what you actually have.

set -uo pipefail

echo "== privileges =="
CAP_EFF="$(grep -m1 '^CapEff:' /proc/self/status | awk '{print $2}')"
echo "  CapEff: $CAP_EFF"

# CAP_SYS_ADMIN is bit 21 -> 0x200000. Mask with python for 64-bit safety.
HAS_ADMIN=$(python3 -c "print(1 if (0x$CAP_EFF >> 21) & 1 else 0)" 2>/dev/null \
            || echo "?")
if [ "$HAS_ADMIN" = "1" ]; then
    echo "  CAP_SYS_ADMIN: YES - mount() may be permitted"
elif [ "$HAS_ADMIN" = "0" ]; then
    echo "  CAP_SYS_ADMIN: no  - mount() will fail with EPERM"
else
    echo "  CAP_SYS_ADMIN: could not determine"
fi
echo "  uid: $(id -u)  ($(id -un))"

echo
echo "== kernel / device support =="
[ -e /dev/fuse ] && echo "  /dev/fuse: present (FUSE mounts may work)" \
                 || echo "  /dev/fuse: missing (no sshfs/rclone/s3fs mounts)"
grep -qw nfs /proc/filesystems 2>/dev/null && echo "  nfs in /proc/filesystems: yes" \
                                           || echo "  nfs in /proc/filesystems: no (module not loaded)"

echo
echo "== client tools =="
for t in mount.nfs mount.cifs sshfs rclone s3fs; do
    if command -v "$t" >/dev/null 2>&1; then
        echo "  $t: $(command -v "$t")"
    else
        echo "  $t: not installed"
    fi
done

echo
echo "== what storage does this platform actually use? =="
FOUND_BACKEND=""
while read -r src mnt fstype opts _; do
    case "$fstype" in
        nfs|nfs4|ceph|cephfs|glusterfs)
            echo "  $fstype at $mnt"
            echo "      source: $src"
            echo "      opts:   $opts"
            FOUND_BACKEND="$fstype" ;;
    esac
done < /proc/mounts
[ -n "$FOUND_BACKEND" ] || echo "  no network filesystem mounted here"
[ -d /etc/ceph ] && echo "  /etc/ceph exists: $(ls /etc/ceph 2>/dev/null | tr '\n' ' ')"

echo
echo "== live test =="
TESTDIR="$(mktemp -d)"
if mount -t tmpfs -o size=1M tmpfs "$TESTDIR" 2>/tmp/.mnterr; then
    umount "$TESTDIR"; rmdir "$TESTDIR"
    echo "  mounting a tmpfs SUCCEEDED - this container can mount things."
    echo "  A NAS mount may work if the server is reachable and tools exist."
    VERDICT=maybe
else
    rmdir "$TESTDIR" 2>/dev/null
    echo "  mounting a tmpfs FAILED: $(tr -d '\n' </tmp/.mnterr)"
    echo "  This container cannot mount anything, regardless of tooling."
    VERDICT=no
fi
rm -f /tmp/.mnterr

echo
if [ "$VERDICT" = "no" ]; then
    cat <<'EOF'
== verdict: you cannot add storage from inside ==

Storage has to be attached when the pod is created, which is a platform-side
operation. Two ways that do NOT require rebuilding from scratch:

  1. Stop the Workshop (关机, not 释放/release), edit it in Aladdin to add the
     mount, then start it again. The login banner says an image is auto-saved
     on shutdown - 在关机时自动保存镜像 - so a graceful stop keeps your
     container disk. Try this before recreating anything.

  2. If Aladdin will not let you edit a stopped Workshop, create a new one with
     Storage filled in. Your work is reproducible in minutes:
       bash scripts/alaya/00_bootstrap_workshop.sh

Either way the weights are the expensive part, so get the PVC attached before
downloading them.
EOF
else
    echo "== verdict: the mount syscall is permitted =="
    echo
    case "$FOUND_BACKEND" in
        ceph|cephfs)
            cat <<'EOF'
This platform uses CephFS, not NFS. That matters: a CephFS mount needs a client
name AND a secret key, which the platform holds and does not hand to the
container. Without the key you cannot mount another volume by hand, even though
the syscall itself is allowed.

Check whether a key was left where you can read it:

    cat /proc/mounts | grep ceph          # look for a name= option
    ls -l /etc/ceph/ 2>/dev/null          # keyring, if any

If there is no keyring, attach the volume through the platform instead: stop
the Workshop (关机, NOT 释放), add the mount in Aladdin, start it again.
EOF
            ;;
        nfs|nfs4)
            cat <<'EOF'
This platform speaks NFS, which needs no credentials, so a hand mount can work.
Set these from the platform's 存储管理 page first - assigning variables avoids
pasting angle brackets, which bash reads as redirects:

    NFS_SERVER=10.x.x.x
    NFS_EXPORT=/exports/your/volume

    apt-get update && apt-get install -y nfs-common
    mkdir -p /pvc
    mount -t nfs "$NFS_SERVER:$NFS_EXPORT" /pvc
    mount | grep /pvc
EOF
            ;;
        *)
            cat <<'EOF'
No network filesystem is mounted here, so there is nothing to copy settings
from. Get the server address and export path from the platform's 存储管理 page.
Assign them to variables rather than pasting placeholders - bash treats a bare
<word> as a redirect, which is where "No such file or directory" comes from.
EOF
            ;;
    esac
    cat <<'EOF'

Either way, two caveats on a hand-made mount: it does not survive a restart, so
it must be re-applied every session, and it bypasses the platform's quota
accounting. Attaching the volume at Workshop creation remains the durable fix.
EOF
fi
