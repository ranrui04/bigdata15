all:
	nvcc *.cu -I/usr/local/cuda-6.5/samples/common/inc -rdc=true -gencode arch=compute_52,code=sm_52 
#-gencode arch=compute_61,code=sm_61

clean:
	rm *.o
