"""camera/camera_model.py  (Camera Model, Phase I: Intrinsics)

A pinhole camera is fixed by five physical numbers plus a resolution:

    f          physical focal length          (mm)
    Ws, Hs     sensor width, height            (mm)
    Nx, Ny     resolution                      (pixels)
    cx, cy     principal point                 (pixels; default = sensor centre)

From those:

    px, py       pixel pitch      = Ws/Nx, Hs/Ny            (mm / pixel)
    fx, fy       pixel focal length = f/px, f/py             (pixels)
    FOVx, FOVy   full field of view = 2 atan(Ws/2f), 2 atan(Hs/2f)

fx, fy, cx, cy assemble into the classic pinhole intrinsic matrix

    K = [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]

which Phase II (projection.py) uses directly.

CONVENTION: camera frame is X right, Y down, Z forward (out of the lens, into
the scene) -- the common photogrammetry/OpenCV convention. Image coordinates
(u, v) have the origin at the top-left, u right, v down, matching how
matplotlib's `imshow` displays an array by default. This choice matters once
we start rendering synthetic images later in this project.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Camera:
    f: float             # physical focal length (mm)
    Ws: float            # sensor width (mm)
    Hs: float            # sensor height (mm)
    Nx: int              # horizontal resolution (px)
    Ny: int              # vertical resolution (px)
    cx: float = None     # principal point (px); default: sensor centre
    cy: float = None
    name: str = "camera"

    def __post_init__(self):
        if self.cx is None:
            self.cx = self.Nx / 2.0
        if self.cy is None:
            self.cy = self.Ny / 2.0

    # ---- derived quantities ------------------------------------------
    @property
    def px(self):
        """Pixel pitch, horizontal (mm/pixel)."""
        return self.Ws / self.Nx

    @property
    def py(self):
        """Pixel pitch, vertical (mm/pixel)."""
        return self.Hs / self.Ny

    @property
    def fx(self):
        """Focal length in horizontal pixel units."""
        return self.f / self.px

    @property
    def fy(self):
        """Focal length in vertical pixel units."""
        return self.f / self.py

    @property
    def fov_x(self):
        """Full horizontal field of view (radians)."""
        return 2.0 * np.arctan(self.Ws / (2.0 * self.f))

    @property
    def fov_y(self):
        """Full vertical field of view (radians)."""
        return 2.0 * np.arctan(self.Hs / (2.0 * self.f))

    @property
    def angular_scale_x(self):
        """Plate scale: FOV per pixel, horizontal (arcsec/pixel)."""
        return np.degrees(self.fov_x) * 3600.0 / self.Nx

    @property
    def angular_scale_y(self):
        """Plate scale: FOV per pixel, vertical (arcsec/pixel)."""
        return np.degrees(self.fov_y) * 3600.0 / self.Ny

    @property
    def K(self):
        """3x3 pinhole intrinsic matrix."""
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]])

    # ---- reporting -----------------------------------------------------
    def summary(self):
        lines = [
            f"--- {self.name} ---",
            f"Physical focal length: f = {self.f:.2f} mm",
            f"Sensor size:            {self.Ws:.2f} x {self.Hs:.2f} mm",
            f"Resolution:             {self.Nx} x {self.Ny} px",
            f"Principal point:        ({self.cx:.1f}, {self.cy:.1f}) px",
            f"Pixel pitch:            px = {self.px*1e3:.3f} um, "
            f"py = {self.py*1e3:.3f} um",
            f"Pixel focal length:     fx = {self.fx:.2f} px, "
            f"fy = {self.fy:.2f} px",
            f"Horizontal FOV:         {np.degrees(self.fov_x):.4f} deg",
            f"Vertical FOV:           {np.degrees(self.fov_y):.4f} deg",
            f"Angular scale:          {self.angular_scale_x:.4f} arcsec/px "
            f"(x), {self.angular_scale_y:.4f} arcsec/px (y)",
        ]
        return "\n".join(lines)


def main():
    # a photographic camera: full-frame mirrorless body, 50 mm lens
    photo = Camera(f=50.0, Ws=36.0, Hs=24.0, Nx=6000, Ny=4000, name="50mm full-frame camera")

    # an amateur astrophotography rig: small refractor + CMOS camera
    astro = Camera(f=2000.0, Ws=13.2, Hs=8.8, Nx=5496, Ny=3672,
                   name="2000mm scope + astro CMOS camera")

    print("=" * 70)
    print("CAMERA MODEL -- PHASE I: INTRINSICS")
    print("=" * 70)
    for cam in (photo, astro):
        print(cam.summary())
        print()

    print("K (astro camera) =")
    print(astro.K)
    print()
    print("Same physics, wildly different scale: the photo lens covers")
    print(f"{np.degrees(photo.fov_x):.1f} deg with {photo.angular_scale_x:.2f} "
          f"arcsec in every pixel, while the telescope covers only")
    print(f"{np.degrees(astro.fov_x)*60:.1f} arcmin but resolves down to "
          f"{astro.angular_scale_x:.3f} arcsec/pixel -- the plate scale that")
    print("matters once we start placing stars on a synthetic sky.")
    print("=" * 70)


if __name__ == "__main__":
    main()
