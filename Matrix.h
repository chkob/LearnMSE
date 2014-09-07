#ifndef	MATRIX_H_INCLUDED
#define MATRIX_H_INCLUDED

#include <stdio.h>
#include <math.h>
#include <stddef.h> 
#include <assert.h>
#include <string.h>
#include <cstdlib>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "cuda_runtime.h"
#include "Utilities.h"
#include "CImg.h"

using namespace std;
using namespace cimg_library;

#define GpuErrorCheck(ans) { PrintError((ans), __FILE__, __LINE__); }
#define MAX_LAYER_NUMBER 10							// Maximum number of layers a network can have
#define CUDA_MAX_NUM_BLOCKS 16384					// Maximum number of cuda blocks in each dimension
#define NUM_THREAD_BLOCK_X 512						// The number of threads in a block in the x dimenison
#define NUM_THREAD_BLOCK_XY 32						// The number of threads in a block in x and y dimensions (for square shaped matrices)
#define EPSILON 1.0e-7
#define BUFFER_SIZE 1000


enum FUNC 
{ 

	SIGMOID = 0,
	SIGMOID_DERIV = 1,
	REC_LINEAR = 2,
	REC_LINEAR_DERIV = 3,
	TANH = 4,
	TANH_DERIV = 5,
	LINEAR = 6,
	LINEAR_DERIV = 7,
	TAN_SIG = 8,
	TAN_SIG_DERIV = 9

};

template<class T>
class Matrix {

public:

	//***** CONSTRUCTORS AND DESTRUCTOR *****// 

	Matrix();
	Matrix(int width, int height);
	Matrix(int width, int height, int depth);
	Matrix(const Matrix<T>& A);
	~Matrix();


	//***** OVERLOADED OPERATORS *****//
	void operator*=(T alpha);
	void operator-=(T alpha);
	void operator=(Matrix<T> B);

	//***** UTILITY FUNCTIONS *****//
	void SetEqualTo(Matrix<T>& B);
	


	Matrix<T> crop(size_t cropTop, size_t cropBottom, size_t cropLeft, size_t cropRight);


	// Initialize all elements of matrix to specified val
	void initializeToVal(T val);

	// Initialize elements of matrix to be random values between the min and max
	void initializeToRandom(T randMin, T randMax);

	void display(char* title);
	void display(char* title, int depthInd);

	void saveToFile(char* filename);
	void saveToFile(char* filename, int depthInd);

	void DeviceToHost(Matrix<T>& hostMat);
	void HostToDevice(Matrix<T>& deviceMat);
	void DeviceToHost();
	void HostToDevice();
	void ImgRead(char *imgName);
	void SetToZero();
	void Reshape(int width, int height, int depth);
	void Insert(Matrix<float>& A, int k);

	Matrix<T> Mat2Block(int blockSize);
	Matrix<T> Sub(Matrix<float>& A);
	Matrix<T> MultSum(Matrix<float>& A);
	Matrix<T> Mult(Matrix<float>& A);
	Matrix<T> Append(Matrix<float>& A);


	//***** GETTERS AND SETTERS *****//
	
	__host__ __device__ int getWidth() const;
	__host__ __device__ int getHeight() const;
	__host__ __device__ int getDepth() const;
	__host__ __device__ int getStride() const;
	__host__ __device__ T* getElements() const;
	
	bool getIsCudaMat() const;
	int getIndex(int x, int y);
	T getElement(int x, int y);
	T getElement(int index);
	void setWidth(int width);
	void setHeight(int height);
	void setDepth(int depth);
	void setIsCudaMat(bool isCudaMat);
	void setElement(int index, T element);
	void setElement(int row, int col, T element);
	void setElements(T* elements);
	void AllocateData(bool isCudaMat);

private:

	//***** DATA *****//

	int width;
	int height;
	int depth;
	T* elements;
	bool isCudaMat;

};


template<class T>
Matrix<T>::Matrix() {
	this->width = 0;
	this->height = 0;
	this->depth = 0;
	this->elements = NULL;
	this->isCudaMat = false;
}

template<class T>
Matrix<T>::Matrix(int width, int height) {

	this->width = width;
	this->height = height;
	this->depth = 1;
	this->elements = NULL;
	this->isCudaMat = false;

}


template<class T>
Matrix<T>::Matrix(int width, int height, int depth) {

	this->width = width;
	this->height = height;
	this->depth = depth;
	this->elements = NULL;
	this->isCudaMat = false;

}

template<class T>
void Matrix<T>::operator=(Matrix<T> B) 
{
	assert(!B.getIsCudaMat());
	if(this == &B) {
		return;
	}
	
	if(this->width != B.getWidth() || this->height != B.getHeight() || this->depth != B.getDepth()) 
	{

		this->~Matrix();
		this->width = B.getWidth(); 
		this->height = B.getHeight();
		this->depth = B.getDepth();
		this->isCudaMat = B.getIsCudaMat();
	}

	if (B.elements != NULL)
	{
		this->elements = new T[this->width * this->height * this->depth];
		memcpy(this->elements, B.getElements(), this->width * this->height * this->depth * sizeof(T)); 
	}

}

template<class T>
void Matrix<T>::operator*=(T a) 
{
	assert(!this->isCudaMat);
	
	
	for (int i = 0; i < this->width * this->height * this->depth; i++)
	{
		this->elements[i] *= a;
	}

}

template<class T>
void Matrix<T>::operator-=(T a) 
{
	assert(!this->isCudaMat);
	
	
	for (int i = 0; i < this->width * this->height * this->depth; i++)
	{
		this->elements[i] -= a;
	}

}

template<class T>
void Matrix<T>::SetToZero() 
{
	assert(this->elements != NULL);
	
	if(isCudaMat)
	{
		int size = this->width * this->height * this->depth * sizeof(T);
		GpuErrorCheck(cudaMemset(this->elements, 0, size));
	}
	else if(!isCudaMat)
	{
		int size = this->width * this->height * this->depth * sizeof(T);
		memset(this->elements, 0, size);
	}
}

template<class T>
void Matrix<T>::Reshape(int width, int height, int depth) 
{
	assert(this->width * this->height * this->depth == width * height * depth);
	
	this->width = width;
	this->height = height;
	this->depth = depth;
}

template<class T>
void Matrix<T>::AllocateData(bool isCudaMat) 
{

	this->isCudaMat = isCudaMat;

	// First free old memory 
	if(elements != NULL) 
	{
		if(!isCudaMat)
		{
			delete[] elements;
			elements = NULL;
		}
		else
		{
			cudaFree(elements);
			elements = NULL;
		}
	}

	// Allocate memory and transfer elements over
	if(!isCudaMat) 
	{
		int size = this->width * this->height * this->depth * sizeof(T);
		this->elements = new T[size];
	} 
	else 
	{
		int size = this->width * this->height * this->depth * sizeof(T);
		GpuErrorCheck(cudaMalloc(&(this->elements), size)); 
	}

}

template<class T>
Matrix<T>::Matrix(const Matrix<T>& A) {

	this->width = A.getWidth();
	this->height = A.getHeight();
	this->depth = A.getDepth();
	this->isCudaMat = A.getIsCudaMat();
	this->elements = NULL;

	if(!isCudaMat && A.getElements() != NULL) 
	{
		this->elements = new T[this->width * this->height * this->depth];
		memcpy(this->elements, A.getElements(), this->width * this->height * this->depth * sizeof(T));
	} 
	else if(A.getElements() != NULL) 
	{
		int size = this->width * this->height * this->depth * sizeof(T);
		GpuErrorCheck(cudaMalloc(&(this->elements), size)); 
		GpuErrorCheck(cudaMemcpy(this->elements, A.getElements(), size, cudaMemcpyDeviceToDevice));
	}

} 

template<class T>
Matrix<T>::~Matrix() 
{

	if(elements != NULL) 
	{
		if(!isCudaMat)
		{
			delete[] elements;
			elements = NULL;
		}
		else
		{
			cudaFree(elements);
			elements = NULL;
		}
	}
	
}


template<class T>
void Matrix<T>::SetEqualTo(Matrix<T>& B) {

	if(this == &B) {
		return ;
	}

	bool sameFormat = (isCudaMat == B.getIsCudaMat());

	if(this->width != B.getWidth() || this->height != B.getHeight() || this->depth != B.getDepth() || !sameFormat) {
		this->~Matrix();
		this->width = B.getWidth(); 
		this->height = B.getHeight();
		this->depth = B.getDepth();
		this->isCudaMat = B.getIsCudaMat();
		if(!isCudaMat) {
			this->elements = new T[this->width * this->height * this->depth];
		} else {
			GpuErrorCheck(cudaMalloc(&(this->elements), this->width * this->height * this->depth * sizeof(T))); 
		}
	}

	assert(isCudaMat == B.getIsCudaMat());

	if(!isCudaMat) {
		memcpy(this->elements, B.getElements(), this->width * this->height * this->depth * sizeof(T)); 
	} else {
		GpuErrorCheck(cudaMemcpy(this->elements, B.getElements(), this->width * this->height * this->depth * sizeof(T), cudaMemcpyDeviceToDevice));
	}

}


template<class T>
void Matrix<T>::initializeToVal(T val) {

	Matrix<T> temp(*this);
	if(isCudaMat) {
		this->deviceToHost(temp);
	}
	T* tempElements = temp.getElements();
	for(int i = 0; i < this->height; i++) {
		for(int j = 0; j < this->width; j++) {
			for(int k = 0; k < this->depth; k++) {

				int index = k * this->width * this->height + i * this->width + j;
				tempElements[index] = val;
			}
		}
	}

	if(!isCudaMat) {
		memcpy(this->elements, tempElements, this->width * this->height * this->depth * sizeof(T));
	} else {
		cudaMemcpy(this->elements, tempElements, this->width * this->height * this->depth * sizeof(T), cudaMemcpyHostToDevice);
	}

}

template<class T>
void Matrix<T>::initializeToRandom(T randMin, T randMax) {

	assert(!isCudaMat);
	for(int i = 0; i < this->height; i++) {
		for(int j = 0; j < this->width; j++) {

			int index = i * this->width + j;
			this->elements[index] = randMin + (randMax - randMin) * rand()/T(RAND_MAX);

		}
	}
}



template<class T>
void Matrix<T>::display(char* title) {

	CImg<T> img(this->width, this->height, 1, this->depth);

	if(!isCudaMat) 
	{
		memcpy(img.data(), this->elements, this->width * this->height * this->depth * sizeof(T));
	} 
	else 
	{
		cudaMemcpy(img.data(), this->elements, this->width * this->height * this->depth * sizeof(T), cudaMemcpyDeviceToHost);
	}
	img.display(title);

}

template<class T>
void Matrix<T>::display(char* title, int depthInd) {

	assert(!isCudaMat);
	CImg<T> img(this->width, this->height);
	memcpy(img.data(), this->elements + depthInd * this->width * this->height, this->width * this->height * sizeof(T));
	img.display(title);

}

template<class T>
void Matrix<T>::saveToFile(char* fileName) {

	assert(!isCudaMat);

	FILE* fp = OpenFile(fileName, "wb");

	fwrite(&this->width, sizeof(int), 1, fp);
	fwrite(&this->height, sizeof(int), 1, fp);
	fwrite(&this->depth, sizeof(int), 1, fp);
	fwrite(this->elements, sizeof(T), this->width * this->height * this->depth, fp);

	fclose(fp);

}

template<class T>
void Matrix<T>::saveToFile(char* fileName, int depthInd) 
{

	assert(!isCudaMat);
	assert(depthInd < this->depth);

	FILE* fp = OpenFile(fileName, "wb");

	int depth = 1;

	fwrite(&this->width, sizeof(int), 1, fp);
	fwrite(&this->height, sizeof(int), 1, fp);
	fwrite(&depth, sizeof(int), 1, fp);
	fwrite(this->elements + this->width * this->height * depthInd, sizeof(T), this->width * this->height, fp);

	fclose(fp);

}

template<class T>
int Matrix<T>::getWidth() const {
	return width;
}

template<class T>
int Matrix<T>::getHeight() const {
	return height;
}

template<class T>
int Matrix<T>::getDepth() const {
	return depth;
}

template<class T>
int Matrix<T>::getIndex(int x, int y) {
	assert(x >= 0 && x < width);
	assert(y >= 0 && y < height);
	return y * width + x;
}

template<class T>
T* Matrix<T>::getElements() const {
	return elements;
}

template<class T>
T Matrix<T>::getElement(int x, int y) {
	assert(this->depth == 1);
	int index = getIndex(x, y);
	assert(index < width * height);
	return getElement(index);
}


template<class T>
T Matrix<T>::getElement(int index) {
	assert(!isCudaMat);
	return elements[index];
}

template<class T>
bool Matrix<T>::getIsCudaMat() const {
	return isCudaMat;
}

template<class T>
void Matrix<T>::setWidth(int width) {
	this->width = width;
}

template<class T>
void Matrix<T>::setHeight(int height) {
	this->height = height;
}


template<class T>
void Matrix<T>::setDepth(int depth) {
	this->depth = depth;
}

template<class T>
void Matrix<T>::setElement(int index, T element) {
	this->elements[index] = element;
}

template<class T>
void Matrix<T>::setElement(int row, int col, T element) {
	int index = row * this->width + col;
	this->setElement(index, element);
}

template<class T>
void Matrix<T>::setElements(T* elements) {
	this->~Matrix();
	this->elements = elements;
} 

template<class T>
void Matrix<T>::setIsCudaMat(bool isCudaMat) {
	this->isCudaMat = isCudaMat;
}

template<class T>
void Matrix<T>::DeviceToHost(Matrix<T>& hostMat) 
{

	assert(isCudaMat);
	assert(hostMat.getWidth() == this->width && hostMat.getHeight() == this->height && hostMat.getDepth() == this->depth);
	GpuErrorCheck(cudaMemcpy(hostMat.getElements(), this->elements, this->width * this->height * this->depth * sizeof(T), cudaMemcpyDeviceToHost));

}

template<class T>
void Matrix<T>::HostToDevice(Matrix<T>& deviceMat) 
{

	assert(!isCudaMat);
	assert(deviceMat.getWidth() == this->width && deviceMat.getHeight() == this->height && deviceMat.getDepth() == this->depth);
	GpuErrorCheck(cudaMemcpy(deviceMat.getElements(), this->elements, this->width * this->height * this->depth * sizeof(T), cudaMemcpyHostToDevice));

}

template<class T>
void Matrix<T>::DeviceToHost() 
{
	assert(isCudaMat);
	
	T* destElements = new T[this->width * this->height * this->depth];
	GpuErrorCheck(cudaMemcpy(destElements, this->elements, this->width * this->height * this->depth * sizeof(T), cudaMemcpyDeviceToHost));
	
	cudaFree(this->elements);
	this->elements = destElements;
	this->isCudaMat = false;
	

}

template<class T>
void Matrix<T>::HostToDevice() 
{

	assert(!isCudaMat);
	
	T* destElements;
	GpuErrorCheck(cudaMalloc(&(destElements), this->width * this->height * this->depth * sizeof(T))); 
	GpuErrorCheck(cudaMemcpy(destElements, this->elements, this->width * this->height * this->depth * sizeof(T), cudaMemcpyHostToDevice));
	
	delete[] this->elements;
	this->elements = destElements;
	this->isCudaMat = true;

}

template<class T>
void Matrix<T>::ImgRead(char* imgName) 
{
	CImg<T> img;
	img.load(imgName);
	img *= 1/255.0f;

	this->width = img.width(); this->height = img.height(); this->depth = 1;
	this->elements = new T[this->width * this->height * this->depth * sizeof(T)];
	memcpy(this->elements, img.data(), this->width * this->height * this->depth * sizeof(T));

}


template<class T>
Matrix<T> Matrix<T>::Mat2Block(int blockSize) 
{
	assert(this->width % blockSize == 0 & this->height % blockSize == 0);

	int sizeBlk = blockSize * blockSize;
	int numBlkW = this->width / blockSize;
	int numBlkH = this->height / blockSize;

	Matrix<T> blockMat(1, sizeBlk);
	blockMat.AllocateData(false);

	int j = floor(rand()/float(RAND_MAX + 1) * numBlkW * numBlkH);

	for (int i = 0; i < sizeBlk; i++)
	{
		int iw = i % blockSize;
		int ih = floor(i / blockSize);
		int jw = j % numBlkW;
		int jh = floor(j / numBlkW);

		int ind = jh * blockSize * this->width + ih * this->width + jw * blockSize + iw;
		blockMat.setElement(i, this->elements[ind]);
	}
	return blockMat;
}


//template<class T>
//Matrix<T> Matrix<T>::Mat2Block(int blockSize) 
//{
//	assert(this->width % blockSize == 0 & this->height % blockSize == 0);
//
//	int sizeBlk = blockSize * blockSize;
//	int numBlkW = this->width / blockSize;
//	int numBlkH = this->height / blockSize;
//
//	Matrix<T> blockMat(numBlkW * numBlkH, sizeBlk);
//	blockMat.AllocateData(false);
//
//	for (int j = 0; j < numBlkW * numBlkH; j++)
//	{
//		for (int i = 0; i < sizeBlk; i++)
//		{
//			int iw = i % blockSize;
//			int ih = floor(i / blockSize);
//			int jw = j % numBlkW;
//			int jh = floor(j / numBlkW);
//
//			int ind = jh * blockSize * this->width + ih * this->width + jw * blockSize + iw;
//			blockMat.setElement(j + i * numBlkW * numBlkH, this->elements[ind]);
//		}
//	}
//	return blockMat;
//}


template<class T>
Matrix<T> Matrix<T>::Sub(Matrix<float>& A) 
{
	assert(this->height  == A.getHeight());
	assert(A.getWidth() == 1);

	Matrix<T> output = Matrix<T> (this->width, this->height);
	output.AllocateData(false);

	float *outputPt = output.getElements();
	float *APt = A.getElements();

	for (int j = 0; j < this->height; j++)
	{
		for (int i = 0; i < this->width; i++)
		{
			int ind = j * this->width + i;
			outputPt[ind] = this->elements[ind] - APt[j];
		}
	}
	return output;
}


template<class T>
Matrix<T> Matrix<T>::MultSum(Matrix<float>& A) 
{
	assert(this->height  == A.getHeight());
	assert(A.getWidth() == 1);

	Matrix<T> output = Matrix<T> (this->width, 1);
	output.AllocateData(false);
	output.SetToZero();

	float *outputPt = output.getElements();
	float *APt = A.getElements();

	for (int j = 0; j < this->height; j++)
	{
		for (int i = 0; i < this->width; i++)
		{
			int ind = j * this->width + i;
			outputPt[i] += this->elements[ind] * APt[j];
		}
	}
	return output;
}

template<class T>
Matrix<T> Matrix<T>::Mult(Matrix<float>& A) 
{
	Matrix<T> output = Matrix<T> (this->width, A.getHeight());
	output.AllocateData(false);

	float *outputPt = output.getElements();
	float *APt = A.getElements();

	for (int j = 0; j < A.getHeight(); j++)
	{
		for (int i = 0; i < this->width; i++)
		{
			int ind = j * this->width + i;
			outputPt[ind] = this->elements[i] * APt[j];
		}
	}
	return output;
}

template<class T>
Matrix<T> Matrix<T>::Append(Matrix<float>& A) 
{
	if (this->height == 0)
		this->height = A.getHeight();
	assert(this->height  == A.getHeight());

	Matrix<T> output = Matrix<T> (this->width + A.getWidth(), this->height);
	output.AllocateData(false);

	float *outputPt = output.getElements();
	float *APt = A.getElements();

	for (int j = 0; j < this->height; j++)
	{
		for (int i = 0; i < this->width; i++)
		{
			int ind = j * output.getWidth() + i;
			int ind1 = j * this->width + i;
			outputPt[ind] = this->elements[ind1];
		}
	}

	for (int j = 0; j < this->height; j++)
	{
		for (int i = this->width; i < output.getWidth(); i++)
		{
			int ind = j * output.getWidth() + i;

			int ind1 = j * A.getWidth() + i - this->width;
			outputPt[ind] = APt[ind1];
		}
	}

	return output;
}

template<class T>
void Matrix<T>::Insert(Matrix<float>& A, int k) 
{
	assert(this->height  == A.getHeight());
	assert(A.getWidth() == 1);

	float *APt = A.getElements();

	for (int j = 0; j < this->height; j++)
	{
		int ind = j * this->width + k;
		this->elements[ind] = APt[j];
	}
}


template<class T>
Matrix<T> Matrix<T>::crop(size_t cropTop, size_t cropBottom, size_t cropLeft, size_t cropRight) 
{// FIXME

	assert(!isCudaMat);
	assert(this->width > (cropLeft + cropRight));
	assert(this->height > (cropTop + cropBottom));
	size_t cropWidth = this->width - (cropLeft + cropRight);
	size_t cropHeight = this->height - (cropTop + cropBottom);
	size_t cropDepth = this->depth;
	Matrix<T> croppedMat(cropWidth, cropHeight, cropDepth);
	croppedMat.AllocateData(false);
	T* croppedElements = croppedMat.getElements();

	size_t startX = cropLeft;
	size_t startY = cropTop;
	size_t startZ = 0;
	size_t endX = this->width - cropRight;
	size_t endY = this->height - cropBottom;
	size_t endZ = this->depth;

	assert(endY != -1);
	size_t index = 0;
	for (size_t k = startZ; k < endZ; k++) {
		for(size_t i = startY; i < endY; i++) {
			for(size_t j = startX; j < endX; j++) {
				size_t pixelIndex = k * this->width * this->height + i * this->width + j;
				croppedElements[index] = this->elements[pixelIndex];
				index++;
			}
		}
	}
	assert(index == (croppedMat.getWidth() * croppedMat.getHeight() * croppedMat.getDepth()));

	return croppedMat;

}



#endif