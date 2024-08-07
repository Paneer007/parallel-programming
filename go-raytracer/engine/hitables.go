package Engine

import (
	"math"
	"math/rand/v2"
)

// HitRecord keeps track of the information of intersection of a ray
// with an object in the scene
type HitRecord struct {
	x         float32
	p, normal Vec3
	material  Material
}

// Interface for a struct that can be hit by a ray
// Defines hit method
type Hittable interface {
	hit(r Ray, t_min, t_max float32, rec *HitRecord) bool
}

// List of multiple Hittable objects
type HittableList []Hittable

// Iterates through list of Hittable objects and applies the hit method
func (h HittableList) hit(r Ray, t_min, t_max float32, rec *HitRecord) bool {
	temp_hit_rec := get_new_hit_record()
	has_hit_anything := false
	closest_t := t_max

	// Iterating through each Hittable object
	for i := 0; i < len(h); i++ {
		if h[i].hit(r, t_min, closest_t, &temp_hit_rec) {
			has_hit_anything = true
			closest_t = temp_hit_rec.x
			*rec = temp_hit_rec
		}
	}
	return has_hit_anything
}

func Color(r Ray, world Hittable, depth int) Vec3 {
	hit_rec := get_new_hit_record()

	zero_vec := Get_Vec3(0,0,0)

	if world.hit(r, 0.001, math.MaxFloat32, &hit_rec) {

		scattered_ray := Get_Ray(zero_vec, zero_vec)
		attenuation_ray := zero_vec

		if depth < 5 && hit_rec.material.Scatter(r, &hit_rec, &attenuation_ray, &scattered_ray) {
			return attenuation_ray.Prod(Color(scattered_ray, world, depth+1))
		} else {
			return zero_vec
		}
	} else {
		unit_dir := r.direction_vector().Normalize()
		t := 0.5 * (unit_dir.Y() + 1.0)
		return Get_Vec3(1.0, 1.0, 1.0).Scalar_mult(1.0 - t).Add(Get_Vec3(0.5, 0.7, 1.0).Scalar_mult(t))
	}
}

func get_new_hit_record() HitRecord {
	zero_vec := Get_Vec3(0,0,0)
	return HitRecord{
		x:      -1.0,
		p:      zero_vec,
		normal: zero_vec,
		material: Material{
			mat_type:    NullMaterial,
			albedo: zero_vec,
		},
	}
}

func (mat Material) Scatter(r Ray, hit_rec *HitRecord, attenuation_ray *Vec3, scattered_ray *Ray) bool {
	switch mat.mat_type {

	case Lambertian_material:
		
		target_vector := hit_rec.p.Add(hit_rec.normal).Add(Random_sphere_point())
		*scattered_ray = Get_Ray(hit_rec.p, target_vector.Sub(hit_rec.p))
		*attenuation_ray = mat.albedo
		return true

	case Metal_material:
		
		reflected_vector := r.direction_vector().Normalize().Reflect(hit_rec.normal)
		*scattered_ray = Get_Ray(hit_rec.p, reflected_vector.Add(Random_sphere_point().Scalar_mult(mat.fuzz)))
		*attenuation_ray = mat.albedo
		return scattered_ray.direction_vector().Dot(hit_rec.normal) > 0

	case Dielectric_material:
		
		reflected_vector := r.direction_vector().Reflect(hit_rec.normal)
		ni_over_nt := float32(0.0)
		*attenuation_ray = Get_Vec3(1, 1, 1)
		refracted_ray := Get_Vec3(0, 0, 0)

		var reflect_prob float32
		var cosine_angle float32
		var outward_normal_vector Vec3


		if r.direction_vector().Dot(hit_rec.normal) > 0 {
			outward_normal_vector = hit_rec.normal.Scalar_mult(-1.0)
			ni_over_nt = mat.ref_idx
			cosine_angle = mat.ref_idx * r.direction_vector().Dot(hit_rec.normal) / r.direction_vector().Norm()
		} else {
			outward_normal_vector = hit_rec.normal
			ni_over_nt = float32(1.0 / mat.ref_idx)
			cosine_angle = -1.0 * r.direction_vector().Dot(hit_rec.normal) / r.direction_vector().Norm()
		}

		if r.direction_vector().Refract(outward_normal_vector, ni_over_nt, &refracted_ray) {
			reflect_prob = schlick_approx(cosine_angle, mat.ref_idx)
		} else {
			*scattered_ray = Get_Ray(hit_rec.p, reflected_vector)
			reflect_prob = 1.0
		}

		if rand.Float32() < reflect_prob {
			*scattered_ray = Get_Ray(hit_rec.p, reflected_vector)
		} else {
			*scattered_ray = Get_Ray(hit_rec.p, refracted_ray)
		}
		return true

	default:
		return true
	}
}
