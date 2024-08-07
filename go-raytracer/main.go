package main

import (
	"fmt"
	"math/rand"
	"sync"

	"example.com/go-raytracer/v2/config"
	Engine "example.com/go-raytracer/v2/engine"
)


func concurrent_rendering(tile_x , tile_y , tile_h , tile_w , ns, nx, ny int, cam Engine.Camera, world Engine.Hittable, buf [][]Engine.Vec3 ){
	// Create a waitgroup whose capacity is the no of pixels being rendered
	var wg sync.WaitGroup
	wg.Add(tile_x * tile_y)

	for j := 0; j < tile_y; j++ {
		for i := 0; i < tile_x; i++ {
			// Create a goroutine for each point being rendered
			go func(i, j, tile_i, tile_j int, buf *[][]Engine.Vec3) {
				defer wg.Done()
				for y := 0; y < tile_h; y++ {
					for x := 0; x < tile_w; x++ {
						col := Engine.Get_Vec3(0, 0, 0)

						for s := 0; s < ns; s++ {
							u := (float32(i*(tile_i)+x) + rand.Float32()) / float32(nx)
							v := (float32(j*(tile_j)+y) + rand.Float32()) / float32(ny)
							r := cam.Get_ray(u, v)
							col = col.Add(Engine.Color(r, world, 0))
						}

						col = col.Scalar_mult(1.0 / float32(ns)).Gamma(2.0)

						ir := float32(255.99 * col.X())
						ig := float32(255.99 * col.Y())
						ib := float32(255.99 * col.Z())

						(*buf)[j*(tile_j)+y][i*(tile_i)+x] = Engine.Get_Vec3(ir, ig, ib)
					}
				}
			}(i, j, tile_w, tile_h, &buf)
		}
	}

	// Synchronisation point
	wg.Wait()
}

func sequential_rendering(tile_x , tile_y , tile_h , tile_w , ns, nx, ny int, cam Engine.Camera, world Engine.Hittable, buf [][]Engine.Vec3 ){

	for j := 0; j < tile_y; j++ {
		for i := 0; i < tile_x; i++ {
				for y := 0; y < tile_h; y++ {
					for x := 0; x < tile_w; x++ {
						col := Engine.Get_Vec3(0, 0, 0)

						for s := 0; s < ns; s++ {
							u := (float32(i*(tile_w)+x) + rand.Float32()) / float32(nx)
							v := (float32(j*(tile_h)+y) + rand.Float32()) / float32(ny)
							r := cam.Get_ray(u, v)
							col = col.Add(Engine.Color(r, world, 0))
						}

						col = col.Scalar_mult(1.0 / float32(ns)).Gamma(2.0)

						ir := float32(255.99 * col.X())
						ig := float32(255.99 * col.Y())
						ib := float32(255.99 * col.Z())

						(buf)[j*(tile_h)+y][i*(tile_w)+x] = Engine.Get_Vec3(ir, ig, ib)
					}
				}
		}
	}
}



func main() {

	parallel_flag := false

	// Output Image Dimensions
	nx, ny, tile_x, tile_y, tile_w, tile_h, ns := config.Image_config()

	// Sets the config for ppm files
	fmt.Print("P3\n", nx, " ", ny, "\n255\n")

	// Camera Config
	lookfrom, lookat, focus_dist, aperture := config.Camera_config()

	// Obtain Camera
	cam := Engine.Get_Camera(lookfrom, lookat, Engine.Get_Vec3(0, 1, 0), 20, float32(nx)/float32(ny), aperture, focus_dist)

	// Generate array of elements that would be used for rendering the scene 
	world := Engine.Random_scene()

	// Generate empty list for all image points in ppm
	buf := make([][]Engine.Vec3, ny)

	for i := range buf {
		buf[i] = make([]Engine.Vec3, nx)
	}
	

	// Uncomment this block to find time of execution
	// Need to pipe out the output of the program to ppm format
	// Will throw an error in the image if you do not comment the chunk of code 

	// start := time.Now()
	
	if parallel_flag{
		concurrent_rendering(tile_x, tile_y, tile_h, tile_w, ns, nx, ny, cam, world, buf)
	}else{
		sequential_rendering(tile_x, tile_y, tile_h, tile_w, ns, nx, ny, cam, world, buf)
	}
	
	// timeElapsed := time.Since(start)
	
	// fmt.Printf("Generation of scene took %s", timeElapsed)

	// Output the rendered image
	for j := ny - 1; j >= 0; j-- {
		for i := 0; i < nx; i++ {
			color := buf[j][i]
			fmt.Print(int(color.R()), " ",
				int(color.G()), " ",
				int(color.B()), "\n")
		}
	}
}