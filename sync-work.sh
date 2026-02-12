kubectl exec -it cuda-dev -- bash -lc 'mkdir -p /workspace/work'
kubectl cp work cuda-dev:/workspace/
