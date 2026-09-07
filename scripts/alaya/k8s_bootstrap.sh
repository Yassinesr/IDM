#!/usr/bin/env bash
# One-time cluster prep on YOUR LAPTOP (not inside a Workshop), for a *shared*
# (共享型) elastic container cluster. This is what stops Aladdin from failing
# with "Failed to pull image ... please check your registry configuration".
#
# Skip this entirely on a *dedicated* (独享型) cluster: the namespace and the
# pull secret are provisioned for you when access is granted.
#
#   export ALAYA_NAMESPACE=user-yassine
#   export ALAYA_REGISTRY=registry.hd-01.alayanew.com:8443
#   export ALAYA_REGISTRY_USER=...        # SMS'd on registry creation, or
#   export ALAYA_REGISTRY_PASS=...        # avatar > 访问管理 > 镜像仓库 > 重置
#   export KUBECONFIG=/path/to/gpu1-config.json   # "kubeconfig下载" on the cluster page
#   bash scripts/alaya/k8s_bootstrap.sh

set -euo pipefail

: "${ALAYA_NAMESPACE:?set ALAYA_NAMESPACE, e.g. user-yassine}"
: "${ALAYA_REGISTRY:?set ALAYA_REGISTRY, e.g. registry.hd-01.alayanew.com:8443}"
: "${ALAYA_REGISTRY_USER:?set ALAYA_REGISTRY_USER}"
: "${ALAYA_REGISTRY_PASS:?set ALAYA_REGISTRY_PASS}"
: "${KUBECONFIG:?set KUBECONFIG to the downloaded cluster config}"

command -v kubectl >/dev/null || { echo "kubectl not on PATH." >&2; exit 1; }

echo "==> cluster"
kubectl cluster-info

echo "==> namespace $ALAYA_NAMESPACE"
kubectl create namespace "$ALAYA_NAMESPACE" 2>/dev/null \
    || echo "    already exists"

echo "==> pull secret 'regcred' (secrets are namespace-scoped, so it must"
echo "    exist inside $ALAYA_NAMESPACE, not just in default)"
kubectl create secret docker-registry regcred \
    --docker-server="$ALAYA_REGISTRY" \
    --docker-username="$ALAYA_REGISTRY_USER" \
    --docker-password="$ALAYA_REGISTRY_PASS" \
    -n "$ALAYA_NAMESPACE" \
    --dry-run=client -o yaml | kubectl apply -f -

echo "==> attaching regcred to the namespace's default ServiceAccount"
echo "    (every Pod binds to it, so Workshops stop needing a manual"
echo "     imagePullSecret and ImagePullBackOff goes away)"
kubectl patch serviceaccount default \
    -p '{"imagePullSecrets":[{"name":"regcred"}]}' \
    -n "$ALAYA_NAMESPACE"

echo
echo "==> verifying"
kubectl get secrets -n "$ALAYA_NAMESPACE"
kubectl get serviceaccount default -n "$ALAYA_NAMESPACE" -o yaml | grep -A2 imagePullSecrets

cat <<EOF

Done. In VS Code: Aladdin > ENVIRONMENTS > gear (Setting Registry) > blue
pencil > enter the same registry username/password, then create the Workshop
with Namespace = $ALAYA_NAMESPACE.
EOF
