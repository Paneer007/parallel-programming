package Engine

import "math"

type Vec3 struct {
	e [3]float32
}

func (v Vec3) X() float32 {
	return v.e[0]
}

func (v Vec3) Y() float32 {
	return v.e[1]
}

func (v Vec3) Z() float32 {
	return v.e[2]
}

func (v Vec3) R() float32 {
	return v.e[0]
}

func (v Vec3) G() float32 {
	return v.e[1]
}

func (v Vec3) B() float32 {
	return v.e[2]
}

func (v Vec3) Add(other Vec3) Vec3 {
	return Vec3{
		e: [3]float32{
			v.e[0] + other.e[0],
			v.e[1] + other.e[1],
			v.e[2] + other.e[2],
		},
	}
}

func (v Vec3) Prod(other Vec3) Vec3 {
	return Vec3{
		e: [3]float32{
			v.e[0] * other.e[0],
			v.e[1] * other.e[1],
			v.e[2] * other.e[2],
		},
	}
}

func (v Vec3) Sub(other Vec3) Vec3 {
	return Vec3{
		e: [3]float32{
			v.e[0] - other.e[0],
			v.e[1] - other.e[1],
			v.e[2] - other.e[2],
		},
	}
}

func (v Vec3) Dot(other Vec3) float32 {
	return v.e[0]*other.e[0] + v.e[1]*other.e[1] + v.e[2]*other.e[2]
}

func (v Vec3) Cross(other Vec3) Vec3 {
	return Vec3{
		e: [3]float32{
			v.e[1]*other.e[2] - v.e[2]*other.e[1],
			-v.e[0]*other.e[2] + v.e[2]*other.e[0],
			v.e[0]*other.e[1] - v.e[1]*other.e[0],
		},
	}
}

func (v Vec3) Scalar_mult(s float32) Vec3 {
	return Get_Vec3(s*v.X(), s*v.Y(), s*v.Z())
}

func (v Vec3) Norm() float32 {
	return float32(math.Sqrt(float64(v.e[0]*v.e[0] + v.e[1]*v.e[1] + v.e[2]*v.e[2])))
}

func (v Vec3) Normalize() Vec3 {
	l := v.Norm()
	return Vec3{
		e: [3]float32{
			v.e[0] / l,
			v.e[1] / l,
			v.e[2] / l,
		},
	}
}

func (v Vec3) R2() float32 {
	return v.e[0]*v.e[0] + v.e[1]*v.e[1] + v.e[2]*v.e[2]
}

func (v Vec3) Gamma(g float32) Vec3 {
	return Get_Vec3(
		float32(math.Pow(float64(v.e[0]), 1.0/float64(g))),
		float32(math.Pow(float64(v.e[1]), 1.0/float64(g))),
		float32(math.Pow(float64(v.e[2]), 1.0/float64(g))),
	)
}

func (v Vec3) Reflect(n Vec3) Vec3 {
	if n.Norm()-1.0 > 0.001 {
		n = n.Normalize()
	}
	return v.Sub(n.Scalar_mult(2 * v.Dot(n)))
}

func (v Vec3) Refract(n Vec3, ni_over_nt float32, refracted *Vec3) bool {
	uv := v.Normalize()
	dt := uv.Dot(n)
	discriminant := float64(1.0 - (ni_over_nt*ni_over_nt)*(1-dt*dt))

	if discriminant > 0 {
		*refracted = uv.Sub(n.Scalar_mult(dt)).Scalar_mult(ni_over_nt).Sub(n.Scalar_mult(float32(math.Sqrt(discriminant))))
		return true
	} else {
		return false
	}
}


func Get_Vec3(x, y, z float32) Vec3 {
	return Vec3{
		e: [3]float32{x, y, z},
	}
}