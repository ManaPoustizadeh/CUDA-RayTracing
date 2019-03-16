
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <vector>
#include <iostream>
#include <cassert>
#include <algorithm>

#define WIDTH   640			//width of the final image
#define HEIGHT	480			//height of the final image
#define BS 16				//number of threads in a block
#define MAX_RAY_DEPTH 4			// This variable controls the maximum recursion depth

#if defined __linux__ || defined __APPLE__
// "Compiled for Linux
#else
// Windows doesn't define these values by default, Linux does
#define M_PI 3.141592653589793
#endif


template<typename T>
class Vec3
{
public:
	T x, y, z;
	__device__ __host__ Vec3() : x(T(0)), y(T(0)), z(T(0)) {}
	__device__ __host__ Vec3(T xx) : x(xx), y(xx), z(xx) {}
	__device__ __host__ Vec3(T xx, T yy, T zz) : x(xx), y(yy), z(zz) {}
	__device__ __host__ Vec3& normalize()
	{
		T nor2 = length2();
		if (nor2 > 0) {
			T invNor = 1 / sqrt(nor2);
			x *= invNor, y *= invNor, z *= invNor;
		}
		return *this;
	}
	__device__ __host__ Vec3<T> operator * (const T &f) const { return Vec3<T>(x * f, y * f, z * f); }
	__device__ __host__ Vec3<T> operator * (const Vec3<T> &v) const { return Vec3<T>(x * v.x, y * v.y, z * v.z); }
	__device__ __host__ T dot(const Vec3<T> &v) const { return x * v.x + y * v.y + z * v.z; }
	__device__ __host__ Vec3<T> operator - (const Vec3<T> &v) const { return Vec3<T>(x - v.x, y - v.y, z - v.z); }
	__device__ __host__ Vec3<T> operator + (const Vec3<T> &v) const { return Vec3<T>(x + v.x, y + v.y, z + v.z); }
	__device__ __host__ Vec3<T>& operator += (const Vec3<T> &v) { x += v.x, y += v.y, z += v.z; return *this; }
	__device__ __host__ Vec3<T>& operator *= (const Vec3<T> &v) { x *= v.x, y *= v.y, z *= v.z; return *this; }
	__device__ __host__ Vec3<T> operator - () const { return Vec3<T>(-x, -y, -z); }
	__device__ __host__ T length2() const { return x * x + y * y + z * z; }
	__device__ __host__ T length() const { return sqrt(length2()); }
	__device__ __host__ friend std::ostream & operator << (std::ostream &os, const Vec3<T> &v)
	{
		os << "[" << v.x << " " << v.y << " " << v.z << "]";
		return os;
	}
};

typedef Vec3<float> Vec3f;

class Sphere
{
public:
	Vec3f center;                           /// position of the sphere
	float radius, radius2;                  /// sphere radius and radius^2
	Vec3f surfaceColor, emissionColor;      /// surface color and emission (light)
	float transparency, reflection;         /// surface transparency and reflectivity
	__device__ __host__ Sphere(
		const Vec3f &c,
		const float &r,
		const Vec3f &sc,
		const float &refl = 0,
		const float &transp = 0,
		const Vec3f &ec = 0) :
		center(c), radius(r), radius2(r * r), surfaceColor(sc), emissionColor(ec),
		transparency(transp), reflection(refl)
	{ /* empty */
	}
	// Compute a ray-sphere intersection using the geometric solution
	__device__ __host__ bool intersect(const Vec3f &rayorig, const Vec3f &raydir, float &t0, float &t1) const
	{
		Vec3f l = center - rayorig;
		float tca = l.dot(raydir);

		if (tca < 0) return false;
		float d2 = l.dot(l) - tca * tca;

		if (d2 > radius2) return false;
		float thc = sqrt(radius2 - d2);

		t0 = tca - thc;
		t1 = tca + thc;

		return true;
	}
};

__device__ __host__ float mix(const float &a, const float &b, const float &mix)
{
	return b * mix + a * (1 - mix);
}

//[comment]
// This is the main trace function. It takes a ray as argument (defined by its origin
// and direction). We test if this ray intersects any of the geometry in the scene.
// If the ray intersects an object, we compute the intersection point, the normal
// at the intersection point, and shade this point using this information.
// Shading depends on the surface property (is it transparent, reflective, diffuse).
// The function returns a color for the ray. If the ray intersects an object that
// is the color of the object at the intersection point, otherwise it returns
// the background color.
//[/comment]
__device__ __host__ Vec3f trace(
	const Vec3f &rayorig,
	const Vec3f &raydir,
	Sphere* spheres,
	const int &depth,
	int size)
{
	//if (raydir.length() != 1) std::cerr << "Error " << raydir << std::endl;
	float tnear = INFINITY;
	const Sphere* sphere = NULL;
	// find intersection of this ray with the sphere in the scene
	for (unsigned i = 0; i < size; ++i) {
		float t0 = INFINITY, t1 = INFINITY;
		if (spheres[i].intersect(rayorig, raydir, t0, t1)) {
			if (t0 < 0) t0 = t1;
			if (t0 < tnear) {
				tnear = t0;
				sphere = &spheres[i];
			}
		}
	}
	// if there's no intersection return black or background color
	if (!sphere) return Vec3f(2);
	//printf("found an intersection, depth is: %d \n", depth);
	Vec3f surfaceColor = 0; // color of the ray/surfaceof the object intersected by the ray
	Vec3f phit = rayorig + raydir * tnear; // point of intersection
	Vec3f nhit = phit - sphere->center; // normal at the intersection point
	nhit.normalize(); // normalize normal direction
					  // If the normal and the view direction are not opposite to each other
					  // reverse the normal direction. That also means we are inside the sphere so set
					  // the inside bool to true. Finally reverse the sign of IdotN which we want
					  // positive.
	float bias = 1e-4; // add some bias to the point from which we will be tracing
	bool inside = false;
	if (raydir.dot(nhit) > 0) nhit = -nhit, inside = true;
	if ((sphere->transparency > 0 || sphere->reflection > 0) && depth < MAX_RAY_DEPTH) {
		float facingratio = -raydir.dot(nhit);
		// change the mix value to tweak the effect
		float fresneleffect = mix(pow(1 - facingratio, 3), 1, 0.1);
		// compute reflection direction (not need to normalize because all vectors
		// are already normalized)
		Vec3f refldir = raydir - nhit * 2 * raydir.dot(nhit);
		refldir.normalize();
		Vec3f reflection = trace(phit + nhit * bias, refldir, spheres, depth + 1, size);
		Vec3f refraction = 0;
		// if the sphere is also transparent compute refraction ray (transmission)
		if (sphere->transparency) {
			float ior = 1.1, eta = (inside) ? ior : 1 / ior; // are we inside or outside the surface?
			float cosi = -nhit.dot(raydir);
			float k = 1 - eta * eta * (1 - cosi * cosi);
			Vec3f refrdir = raydir * eta + nhit * (eta *  cosi - sqrt(k));
			refrdir.normalize();
			refraction = trace(phit - nhit * bias, refrdir, spheres, depth + 1, size);
		}
		// the result is a mix of reflection and refraction (if the sphere is transparent)
		surfaceColor = (
			reflection * fresneleffect +
			refraction * (1 - fresneleffect) * sphere->transparency) * sphere->surfaceColor;
	}

	else {
		
		// it's a diffuse object, no need to raytrace any further
		for (unsigned i = 0; i < size; ++i) {
			if (spheres[i].emissionColor.x > 0) {
				// this is a light
				Vec3f transmission = 1;
				Vec3f lightDirection = spheres[i].center - phit;
				lightDirection.normalize();
				
				for (unsigned j = 0; j < size; ++j) {
					if (i != j) {
						
						float t0, t1;
						if (spheres[j].intersect(phit + nhit * bias, lightDirection, t0, t1)) {
							transmission = 0;
							break;
						}
						//printf("i = %d, j = %d\n", i, j);
					}
				}
				surfaceColor += sphere->surfaceColor * transmission *
					std::max(float(0), nhit.dot(lightDirection)) * spheres[i].emissionColor;
			}
		}
	}
	//printf("trace ended\n");
	return surfaceColor + sphere->emissionColor;
}


//Here each thread specifies the ray origin and direction then calls the trace function
__global__ void par_render(Sphere* spheres, Vec3f* pixel, float invHeight, float invWidth, float angle, float fov, float aspectRatio, int size){
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	pixel += x + y*WIDTH;
	float xx = (2 * ((x + 0.5) * invWidth) - 1) * angle * aspectRatio;
	float yy = (1 - 2 * ((y + 0.5) * invHeight)) * angle;	
	Vec3f raydir(xx, yy, -1);
	raydir.normalize();
	//printf("entered kernel\n");
	if (x < WIDTH && y < HEIGHT) {
		*pixel = trace(Vec3f(0), raydir, spheres, 0, size);
	}
		
}

//In this function we allocate GPU memories, copy the data to GPU and call the kernel (par_render)
Vec3f parallelRender(std::vector<Sphere> &spheres)
{
	int size = spheres.size();
	unsigned int spheres_size = size * sizeof(Sphere);
	//printf("d_sphere's first element origin: %d", spheres_size);
	//std::vector<Sphere> d_spheres;
	Sphere *d_spheres;
	Vec3f *image = new Vec3f[WIDTH * HEIGHT], *pixel = image;
	//Vec3f *d_image;
	Vec3f *d_pixel;
	unsigned int image_size = WIDTH*HEIGHT * sizeof(Vec3f);
	float invWidth = 1 / float(WIDTH), invHeight = 1 / float(HEIGHT);
	float fov = 30, aspectratio = WIDTH / float(HEIGHT);
	float angle = tan(M_PI * 0.5 * fov / 180.);


	
	cudaError_t err;
	cudaEvent_t start0, stop0;
	dim3 grid((WIDTH-1)/BS+1, (HEIGHT-1)/BS + 1, 1);
	dim3 threads(BS, BS, 1);

	err = cudaEventCreate(&start0);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaEventCreate(&stop0);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaEventRecord(start0, NULL);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	err = cudaMalloc((void **)&d_spheres, spheres_size);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	err = cudaMalloc((void **)&d_pixel, image_size);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	err = cudaMemcpy(d_spheres, spheres.data(), spheres_size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	
	par_render << <grid, threads >> > (d_spheres, d_pixel, invHeight, invWidth, angle, fov, aspectratio, size);

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		//printf("blah blah\n");
		system("PAUSE");
		exit(EXIT_FAILURE);
	}
	err = cudaMemcpy(pixel, d_pixel, image_size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		system("PAUSE");
		exit(EXIT_FAILURE);
	}
	
	err = cudaEventRecord(stop0, NULL);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaEventSynchronize(stop0);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	float elapsed_time = 0.0f;
	err = cudaEventElapsedTime(&elapsed_time, start0, stop0);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	printf("Calculation time in msec = %f\n", elapsed_time);
	// Save result to a PPM image (keep these flags if you compile under Windows)
	std::ofstream ofs("./picture.ppm", std::ios::out | std::ios::binary);
	ofs << "P6\n" << WIDTH << " " << HEIGHT << "\n255\n";
	for (unsigned i = 0; i < WIDTH * HEIGHT; ++i) {
		ofs << (unsigned char)(std::min(float(1), image[i].x) * 255) <<
			(unsigned char)(std::min(float(1), image[i].y) * 255) <<
			(unsigned char)(std::min(float(1), image[i].z) * 255);
	}
	ofs.close();
	cudaFree(d_pixel);
	cudaFree(d_spheres);
	system("PAUSE");

}

//[comment]
// Main rendering function. We compute a camera ray for each pixel of the image
// trace it and return a color. If the ray hits a sphere, we return the color of the
// sphere at the intersection point, else we return the background color.
//[/comment]
/*void render(std::vector<Sphere> &spheres)
{
	unsigned width = 640, height = 480;
	Vec3f *image = new Vec3f[width * height], *pixel = image;
	float invWidth = 1 / float(width), invHeight = 1 / float(height);
	float fov = 30, aspectratio = width / float(height);
	float angle = tan(M_PI * 0.5 * fov / 180.);
	// Trace rays
	for (unsigned y = 0; y < height; ++y) {
		for (unsigned x = 0; x < width; ++x, ++pixel) {
			float xx = (2 * ((x + 0.5) * invWidth) - 1) * angle * aspectratio;
			float yy = (1 - 2 * ((y + 0.5) * invHeight)) * angle;
			Vec3f raydir(xx, yy, -1);
			raydir.normalize();
			*pixel = trace(Vec3f(0), raydir, spheres, 0, spheres.size());
		}
	}
	// Save result to a PPM image (keep these flags if you compile under Windows)
	std::ofstream ofs("./untitled.ppm", std::ios::out | std::ios::binary);
	ofs << "P6\n" << width << " " << height << "\n255\n";
	for (unsigned i = 0; i < width * height; ++i) {
		ofs << (unsigned char)(std::min(float(1), image[i].x) * 255) <<
			(unsigned char)(std::min(float(1), image[i].y) * 255) <<
			(unsigned char)(std::min(float(1), image[i].z) * 255);
	}
	ofs.close();
	delete[] image;
}*/

//[comment]
// In the main function, we will create the scene which is composed of 5 spheres
// and 1 light (which is also a sphere). Then, once the scene description is complete
// we render that scene, by calling the render() function.
//[/comment]
int main(int argc, char **argv)
{
	srand(13);
	std::vector<Sphere> spheres;
	// position, radius, surface color, reflectivity, transparency, emission color
	spheres.push_back(Sphere(Vec3f(0.0, -10004, -20), 10000, Vec3f(0.20, 0.20, 0.20), 0, 0.0));
	spheres.push_back(Sphere(Vec3f(0.0, 0, -20), 4, Vec3f(1.00, 0.32, 0.36), 1, 0.5));
	spheres.push_back(Sphere(Vec3f(5.0, -1, -15), 2, Vec3f(0.90, 0.76, 0.46), 1, 0.0));
	spheres.push_back(Sphere(Vec3f(5.0, 0, -25), 3, Vec3f(0.65, 0.77, 0.97), 1, 0.0));
	spheres.push_back(Sphere(Vec3f(-5.5, 0, -15), 3, Vec3f(0.90, 0.90, 0.90), 1, 0.0));
	// light
	spheres.push_back(Sphere(Vec3f(0.0, 20, -30), 3, Vec3f(0.00, 0.00, 0.00), 0, 0.0, Vec3f(3)));
	parallelRender(spheres);

	return 0;

}