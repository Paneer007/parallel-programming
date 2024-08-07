package Engine

import (
	"math"
	"math/rand/v2"
)

type Sphere struct {
	center   Vec3
	radius   float32
	material Material
}

// Returns the sphere object
func Get_Sphere(center Vec3, radius float32, mat Material) Sphere {
	return Sphere{
		center,
		radius,
		mat,
	}
}

// Checks if a ray intersects a sphere or not
func (s Sphere) hit(r Ray, t_min, t_max float32, rec *HitRecord) bool {
	origin := r.origin_vector().Sub(s.center)

	a := r.direction_vector().R2()
	b := origin.Dot(r.direction_vector())
	c := origin.R2() - s.radius*s.radius
	discriminant := b*b - a*c

	// Check if solution exists, it intersects the sphere
	if discriminant > 0 {
		
		// Quadratic Roots

		// Root 1
		temp := (-b - float32(math.Sqrt(float64(discriminant)))) / a
		
		// Check if it falls in the range
		if temp < t_max && temp > t_min {
			rec.x = temp
			rec.p = r.point_vector(rec.x)
			rec.normal = rec.p.Sub(s.center).Scalar_mult(1.0 / s.radius)
			rec.material = s.material
			return true
		}

		// Root 2
		temp = (-b + float32(math.Sqrt(float64(discriminant)))) / a
		
		// Check if it falls in the range
		if temp < t_max && temp > t_min {
			rec.x = temp
			rec.p = r.point_vector(rec.x)
			rec.normal = rec.p.Sub(s.center).Scalar_mult(1.0 / s.radius)
			rec.material = s.material
			return true
		}
	}

	// It does not intersect sphere, hence we return false
	return false
}

// Returns a random point on the sphere
func Random_sphere_point() Vec3 {
	p := Get_Vec3(1, 1, 1)

	for p.R2() >= 1.0 {
		p = Get_Vec3(rand.Float32(), rand.Float32(), rand.Float32()).Scalar_mult(2.0).Sub(Get_Vec3(1, 1, 1))
	}

	return p
}

// Returns a random point on the sphere
func Random_disk_point() Vec3 {
	p := Get_Vec3(1, 1, 1)

	for p.R2() >= 1.0 {
		p = Get_Vec3(rand.Float32(), rand.Float32(), 0).Scalar_mult(2.0).Sub(Get_Vec3(1, 1, 0))
	}

	return p
}

