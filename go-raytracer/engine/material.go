package Engine


type MaterialType int

const (
	NullMaterial MaterialType = 0
	Lambertian_material   MaterialType = 1
	Metal_material        MaterialType = 2
	Dielectric_material   MaterialType = 3
)

type Material struct {
	mat_type           MaterialType
	albedo        Vec3
	fuzz, ref_idx float32
}

func lambertian_material(ax, ay, az float32) Material {
	return Material{
		mat_type:    Lambertian_material,
		albedo: Get_Vec3(ax, ay, az),
	}
}

func metal_material(ax, ay, az, fuzz float32) Material {
	return Material{
		mat_type:    Metal_material,
		albedo: Get_Vec3(ax, ay, az),
		fuzz:   fuzz,
	}
}

func dielectric_material(ref_idx float32) Material {
	return Material{
		mat_type:     Dielectric_material,
		albedo:  Get_Vec3(1.0, 1.0, 1.0),
		fuzz:    0,
		ref_idx: ref_idx,
	}

}