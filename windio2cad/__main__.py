import argparse
import yaml
import numpy as np
from scipy.interpolate import PchipInterpolator as spline
import windio2cad.geometry_tools as geom
from numpy.linalg import norm
from math import sin, cos
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import savemat


class Blade:
    """
    This class renders one blade for the rotor.
    """

    def __init__(self, yaml_filename: str):
        """
        The constructor opens the YAML file and extracts the blade
        and airfoil information into instance attributes.

        Parameters
        ----------
        yaml_filename: str
            Filename that contains the geometry for the rotor.
        """
        geometry = yaml.load(open(yaml_filename, "r"), yaml.FullLoader)
        self.outer_shape = geometry["components"]["blade"]["outer_shape_bem"]
        self.airfoils = geometry["airfoils"]

    @staticmethod
    def myinterp(xi, x, f) -> np.array:
        myspline = spline(x, f)
        return myspline(xi)

    def generate_lofted(self, n_span_min=100, n_xy=400) -> np.array:
        """
        Creates the lofted shape of a blade and returns a NumPy array
        of the polygons at each cross section.

        Parameters
        ----------
        n_span_min: int
            Number of cross sections to create across span of
            blade.

        n_xy: int
            The number of x, y points in the polygons at each slice of
            the blade.

        Returns
        -------
        np.array
            An array of the polygons at each cross section of the blade.
        """
        # Use yaml grid points and others that we add
        r_span = np.unique(
            np.r_[
                np.linspace(0.0, 1.0, n_span_min),
                self.outer_shape["chord"]["grid"],
                self.outer_shape["twist"]["grid"],
                self.outer_shape["pitch_axis"]["grid"],
                self.outer_shape["reference_axis"]["x"]["grid"],
                self.outer_shape["reference_axis"]["y"]["grid"],
                self.outer_shape["reference_axis"]["z"]["grid"],
            ]
        )
        n_span = len(r_span)

        # Read in blade spanwise geometry values and put on common grid
        chord = self.myinterp(
            r_span,
            self.outer_shape["chord"]["grid"],
            self.outer_shape["chord"]["values"],
        )
        twist = self.myinterp(
            r_span,
            self.outer_shape["twist"]["grid"],
            self.outer_shape["twist"]["values"],
        )
        pitch_axis = self.myinterp(
            r_span,
            self.outer_shape["pitch_axis"]["grid"],
            self.outer_shape["pitch_axis"]["values"],
        )
        ref_axis = np.c_[
            self.myinterp(
                r_span,
                self.outer_shape["reference_axis"]["x"]["grid"],
                self.outer_shape["reference_axis"]["x"]["values"],
            ),
            self.myinterp(
                r_span,
                self.outer_shape["reference_axis"]["y"]["grid"],
                self.outer_shape["reference_axis"]["y"]["values"],
            ),
            self.myinterp(
                r_span,
                self.outer_shape["reference_axis"]["z"]["grid"],
                self.outer_shape["reference_axis"]["z"]["values"],
            ),
        ]

        # Get airfoil names and thicknesses
        af_position = self.outer_shape["airfoil_position"]["grid"]
        af_used = self.outer_shape["airfoil_position"]["labels"]
        n_af_span = len(af_position)
        n_af = len(self.airfoils)
        name = n_af * [""]
        r_thick = np.zeros(n_af)
        for i in range(n_af):
            name[i] = self.airfoils[i]["name"]
            r_thick[i] = self.airfoils[i]["relative_thickness"]

        # Create common airfoil coordinates grid
        coord_xy = np.zeros((n_af, n_xy, 2))
        for i in range(n_af):
            points = np.c_[
                self.airfoils[i]["coordinates"]["x"],
                self.airfoils[i]["coordinates"]["y"],
            ]

            # Check that airfoil points are declared from the TE suction side to TE pressure side
            idx_le = np.argmin(points[:, 0])
            if np.mean(points[:idx_le, 1]) > 0.0:
                points = np.flip(points, axis=0)

            # Remap points using class AirfoilShape
            af = geom.AirfoilShape(points=points)
            af.redistribute(n_xy, even=False, dLE=True)
            af_points = af.points

            # Add trailing edge point if not defined
            if [1, 0] not in af_points.tolist():
                af_points[:, 0] -= af_points[np.argmin(af_points[:, 0]), 0]
            c = max(af_points[:, 0]) - min(af_points[:, 0])
            af_points[:, :] /= c

            coord_xy[i, :, :] = af_points

        # Reconstruct the blade relative thickness along span with a pchip
        r_thick_used = np.zeros(n_af_span)
        coord_xy_used = np.zeros((n_af_span, n_xy, 2))
        coord_xy_interp = np.zeros((n_span, n_xy, 2))
        coord_xy_dim = np.zeros((n_span, n_xy, 2))

        for i in range(n_af_span):
            for j in range(n_af):
                if af_used[i] == name[j]:
                    r_thick_used[i] = r_thick[j]
                    coord_xy_used[i, :, :] = coord_xy[j, :, :]

        r_thick_interp = self.myinterp(r_span, af_position, r_thick_used)

        # Spanwise interpolation of the profile coordinates with a pchip
        r_thick_unique, indices = np.unique(r_thick_used, return_index=True)
        coord_xy_interp = np.flip(
            self.myinterp(
                np.flip(r_thick_interp), r_thick_unique, coord_xy_used[indices, :, :]
            ),
            axis=0,
        )
        for i in range(n_span):
            # Correction to move the leading edge (min x point) to (0,0)
            af_le = coord_xy_interp[i, np.argmin(coord_xy_interp[i, :, 0]), :]
            coord_xy_interp[i, :, 0] -= af_le[0]
            coord_xy_interp[i, :, 1] -= af_le[1]
            c = max(coord_xy_interp[i, :, 0]) - min(coord_xy_interp[i, :, 0])
            coord_xy_interp[i, :, :] /= c
            # If the rel thickness is smaller than 0.4 apply a trailing ege smoothing step
            if r_thick_interp[i] < 0.4:
                coord_xy_interp[i, :, :] = geom.trailing_edge_smoothing(
                    coord_xy_interp[i, :, :]
                )

        # Offset by pitch axis and scale for chord
        coord_xy_dim = coord_xy_interp.copy()
        coord_xy_dim[:, :, 0] -= pitch_axis[:, np.newaxis]
        coord_xy_dim = coord_xy_dim * chord[:, np.newaxis, np.newaxis]

        # Rotate to twist angle
        coord_xy_dim_twisted = np.zeros(coord_xy_interp.shape)
        for i in range(n_span):
            x = coord_xy_dim[i, :, 0]
            y = coord_xy_dim[i, :, 1]
            coord_xy_dim_twisted[i, :, 0] = x * np.cos(twist[i]) - y * np.sin(twist[i])
            coord_xy_dim_twisted[i, :, 1] = y * np.cos(twist[i]) + x * np.sin(twist[i])

        # Assemble lofted shape along reference axis
        lofted_shape = np.zeros((n_span, n_xy, 3))
        for i in range(n_span):
            for j in range(n_xy):
                lofted_shape[i, j, :] = (
                    np.r_[
                        coord_xy_dim_twisted[i, j, 1],
                        coord_xy_dim_twisted[i, j, 0],
                        0.0,
                    ]
                    + ref_axis[i, :]
                )

        return lofted_shape

if __name__ == "__main__":

    # Create a command line parser
    parser = argparse.ArgumentParser(
        description="Translate a yaml definition of a semisubmersible platform into an OpenSCAD source file."
    )
    parser.add_argument("--input", help="Input .yaml file", required=True)
    args = parser.parse_args()

    intermediate_openscad = "intermediate.scad"

    print(f"Input yaml: {args.input}")
    print("Parsing .yaml ...")

    print("Rendering blade only...")
    blade = Blade(args.input)
    lofted_shape = blade.generate_lofted()
    print(lofted_shape)
    print(lofted_shape.shape)
    
    savemat('data.mat', {'lofted_shape': lofted_shape})
    
    # fig = plt.figure(figsize=(10, 6))
    # ax = fig.add_subplot(111, projection='3d')
    # for i in range(lofted_shape.shape[0]):
    #     slice_i = lofted_shape[i]
    #     ax.plot(slice_i[:, 0], slice_i[:, 1], slice_i[:, 2], color='b', alpha=0.3)
    #
    # ax.set_title('Lofted Turbine Blade Outline')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.tight_layout()
    # plt.show()
            
    print("Done!")
