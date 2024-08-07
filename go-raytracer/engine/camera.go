package Engine

import (
	"math"
	"math/rand/v2"
)

// Camera:
// This struct defines the view of the rendered scene to be displayed
type Camera struct {
	o, // Origin
	llc, // Lower left corner
	hor, // horizontal
	ver, // vertical
	u, v, w Vec3
	lr float32 // lens_radius
}

// returns the ray for a point in the scene
func (c Camera) Get_ray(s, t float32) Ray {

	// Construct a camera ray originating from the defocus disk and directed at a randomly
	// sampled point around the pixel location i, j.

	rd := Random_disk_point().Scalar_mult(c.lr) // lens_radius
	offset := c.u.Scalar_mult(rd.X()).Add(c.v.Scalar_mult(rd.Y()))

	direction := c.llc
	direction = direction.Add(c.hor.Scalar_mult(s))
	direction = direction.Add(c.ver.Scalar_mult(t))
	direction = direction.Sub(c.o)
	direction = direction.Sub(offset)

	return Ray{
		C: c.o.Add(offset),
		M: direction,
	}
}

// Returns the camera for this scene
func Get_Camera(P, Q, vup Vec3, vfov, aspect, aperture, focus_dist float32) Camera {
	// P -> point you look at
	// Q -> point you look from
	angle := float64(vfov * math.Pi / 180.0)
	h_h := float32(math.Tan(angle / 2.0))
	h_w := aspect * h_h

	origin := P

	w := P.Sub(Q).Normalize()
	u := vup.Cross(w).Normalize()
	v := w.Cross(u)

	llc := origin.Sub(u.Scalar_mult(h_w * focus_dist))
	llc = llc.Sub(v.Scalar_mult(h_h * focus_dist))
	llc = llc.Sub(w.Scalar_mult(focus_dist))

	horizontal := u.Scalar_mult(2.0 * h_w * focus_dist)
	vertical := v.Scalar_mult(2.0 * h_h * focus_dist)

	return Camera{
		origin,
		llc,
		horizontal,
		vertical,
		u, v, w,
		aperture / 2.0,
	}
}

// Schlick's approximation for reflectance.
func schlick_approx(cosine float32, ref_idx float32) float32 {
	r0 := (1.0 - ref_idx) / (1.0 + ref_idx)
	r0 = r0 * r0
	return r0 + (1.0-r0)*float32(math.Pow(float64(1.0-cosine), 5.0))
}

/* example scenes to render */
func Random_scene() HittableList {
	var world HittableList

	world = append(world, Get_Sphere(Get_Vec3(0, -1000, 0), 1000, lambertian_material(0.5, 0.5, 0.5)))

	for a := -11; a < 12; a++ {
		for b := -11; b < 12; b++ {
			mat_prob := rand.Float32()
			center := Get_Vec3(float32(a)+0.9*rand.Float32(), 0.2, float32(b)+0.9*rand.Float32())

			if center.Sub(Get_Vec3(4, 0.2, 0)).Norm() > 0.7 {
				if mat_prob < 0.8 {
					world = append(world, Get_Sphere(center, 0.3, lambertian_material(
						rand.Float32()*rand.Float32(),
						rand.Float32()*rand.Float32(),
						rand.Float32()*rand.Float32())))
				} else if mat_prob < 0.95 {
					world = append(world, Get_Sphere(center, 0.3, metal_material(
						0.5*(1+rand.Float32()),
						0.5*(1+rand.Float32()),
						0.5*(1+rand.Float32()),
						0.5*(1+rand.Float32()),
					)))
				} else {
					world = append(world, Get_Sphere(center, 0.3, dielectric_material(1.5)))
				}
			}
		}
	}

	world = append(world, Get_Sphere(Get_Vec3(0, 1, 0), 1.0, dielectric_material(1.5)))
	world = append(world, Get_Sphere(Get_Vec3(-4, 1, 0), 1.0, lambertian_material(0.4, 0.2, 0.1)))
	world = append(world, Get_Sphere(Get_Vec3(4, 1, 0), 1.0, metal_material(0.7, 0.6, 0.5, 0.0)))
	return world
}
