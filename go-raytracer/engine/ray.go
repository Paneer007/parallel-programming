package Engine


func Get_Ray(a, b  Vec3) Ray {
	return Ray{
		C: a,
		M: b,
	}
}

func (r Ray) origin_vector() Vec3 { return r.C }

func (r Ray) direction_vector() Vec3 { return r.M }

func (r Ray) point_vector(x float32) Vec3 { return r.origin_vector().Add(r.direction_vector().Scalar_mult(x)) }

type Ray struct{ C, M Vec3 }