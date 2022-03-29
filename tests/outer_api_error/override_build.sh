gcc -shared -fPIC override_veo_memcpy.c -ldl -I/opt/nec/ve/veos/include -o liboverride_veo_memcpy.so
gcc -shared -fPIC override_cuda_memcpy.c -ldl -o liboverride_cuda_memcpy.so
