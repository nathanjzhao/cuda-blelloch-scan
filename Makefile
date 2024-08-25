scan: main.cu scan.o Makefile 
	nvcc -o scan main.cu scan.o 

scan.o: scan.cu
	nvcc -c scan.cu

clean:
	rm -f *.o scan