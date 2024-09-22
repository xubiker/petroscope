from petroscope.segmentation.utils.data import Class, ClassAssociation


class LumenStoneClasses:

    S1v1 = ClassAssociation(
        classes=(
            Class(code=0, label="BG", name="background", color="#000000"),
            Class(code=1, label="Ccp", name="chalcopyrite", color="#ff0000"),
            Class(code=2, label="Gl", name="galena", color="#cbff00"),
            Class(code=4, label="Brt", name="bornite", color="#0065ff"),
            Class(
                code=6,
                label="Py/Mrc",
                name="pyrite/marcasite",
                color="#ff4c4c",
            ),
            Class(code=8, label="Sph", name="sphalerite", color="#4cff93"),
            Class(
                code=11,
                label="Tnt/Ttr",
                name="tenantite-tetrahedrite",
                color="#ff9999",
            ),
        )
    )
