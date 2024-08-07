package config

import Engine "example.com/go-raytracer/v2/engine"


func Camera_config()(Engine.Vec3, Engine.Vec3, float32, float32){
	lookfrom := Engine.Get_Vec3(13, 2, 3)
	lookat := Engine.Get_Vec3(0, 0, 0)
	focus_dist := float32(10.0)
	aperture := float32(0.1)
	return lookfrom, lookat, focus_dist, aperture
}

func Image_config() ( int, int, int, int, int, int, int){

	scale := 0.4
	nx := int(1920 * scale)
	ny := int(1080 * scale)
	tile_x := 4
	tile_y := 2
	tile_w := nx / tile_x
	tile_h := ny / tile_y
	ns := 40

	return  nx, ny, tile_x, tile_y, tile_w, tile_h, ns
}