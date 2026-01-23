import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz
import json

# This is a direct copy from PyMoosh. Maybe PyMoosh should just be a dependency, at this point

import sys

if sys.version_info[0] >= 3 and sys.version_info[1] >= 10:
    from refractiveindex import RefractiveIndexMaterial


def conv_to_nm(length, unit):
    """Converts a length from "unit" to nm, because everything has been coded
    in nm
    """
    if unit == "m":
        return np.array(length) * 1e9
    elif unit == "um":
        return np.array(length) * 1e3
    elif unit == "mm":
        return np.array(length) * 1e6
    elif unit == "pm":
        return np.array(length) * 1e-3
    elif unit == "nm":
        # Just in case we get here but didn't need to
        return np.array(length)
    else:
        print("Please provide lengths in m, mm, um, pm or nm")


class Structure:
    """
    Each instance of Structure describes a structure completely.
    This includes the materials, the dimensions of the structurations and the
    thickness of each layer

    Args:
        layer_type (list) : the list of materials in each layers
        thickness (list) : thickness of each layer in nm
        interfaces (list) : a list of x (and y) interfaces
        homo_layer (list) : a list of whether each layer is homogeneous
        pmls (list) : a list of whether each zone is a pml
        units (str) : the length unit used for thickness

    Example: [1.,'Si','Au'] means the layer will be made of
    air between x=0 and x=interface[1],
    then Silicon between x=interface[1] and x=interface[2],
    then Gold between x=interface[2] and x=interface[3]

    The thickness of each layer is given in the :thickness: list, in nanometers
    by default. The thickness of the superstrate is assumed to be zero by most
    of the routines (like :coefficient:, :absorption:) so that the first
    interface is considered as the phase reference. The reflection coefficient
    of the structure will thus never be impacted by the thickness of the first
    layer.For other routines (like :field:), this thickness define the part
    of the superstrate (or substrate) that must be represented on the figure.

    """

    def __init__(
        self,
        layers,
        thicknesses,
        interfaces,
        homo_layer,
        pmls,
        verbose=True,
        unit="nm",
        si_units=False,
    ):

        if unit != "nm":
            thicknesses = conv_to_nm(thicknesses, unit)
            if not (si_units):
                print(
                    "I can see you are using another unit than nanometers, ",
                    "please make sure you keep using that unit everywhere.",
                    " To suppress this message, add the keyword argument si_units=True when you call Structure",
                )

        self.unit = unit

        self.layers = layers
        self.thicknesses = thicknesses
        self.interfaces = interfaces
        self.period = interfaces[-1]
        # self.interfaces = [t - self.period/2 for t in self.interfaces]
        self.homo_layer = homo_layer
        self.pmls = pmls

    # str not implemented yet
    # def __str__(self):
    #     materials = [str(self.materials[i]) for i in range(len(self.materials))]
    #     s = f"materials: {materials}\nlayers: {self.layer_type}\nthicknesses: {self.thickness}"
    #     return s

    def get_perm_top(self, wavelength):
        return self.layers[0][0].get_permittivity(wavelength)
    
    def get_perm_bot(self, wavelength):
        return self.layers[-1][0].get_permittivity(wavelength)

    def polarizability(self, wavelength):
        # TODO: this will have to change completely, or maybe not even be used
        # """Computes the actual permittivity and permeability of each material considered in
        # the structure. This method is called before each calculation.

        # Args:
        #     wavelength (float): the working wavelength (in nanometers)
        # """

        # # Create empty mu and epsilon arrays
        # mu = np.ones_like(self.materials, dtype=complex)
        # epsilon = np.ones_like(self.materials, dtype=complex)
        # # Loop over all materials
        # for k in range(len(self.materials)):
        #     # Populate epsilon and mu arrays from the material.
        #     material = self.materials[k]
        #     epsilon[k] = material.get_permittivity(wavelength)
        #     mu[k] = material.get_permeability(wavelength)

        return None

    # TODO: implement several checks:
    """
        - that the number of materials is the same in all layers
        - that the number of materials corresponds to the number of interfaces
        - that the layers are made only of materials (maybe accept constant permittivity values also?)
        - that the number of values in pmls corresponds to the number of interfaces
        - compute and save perm_top and perm_bot as the 
           incoming and outgoing material permittivities
           + check that the first and last layer are homogeneous
    """


class Material:
    """
    Types of material (default):
          type                   / format of "mat" variable:

        - material               / Material object
        - CustomFunction         / function (wav)
        - simple_perm            / complex
        - magnetic               / list(complex, float) of size 2
        - Database               / string
              Database types can take many special types

        There are four special types:
        -> when importing from a text file (lambda, n, k format),
            Then the specialType variable should be set to 'File'
            - File                   / file name
        -> when importing from the Refractive Index Database
            Then the specialType variable should be set to "RII"
            - RefractiveIndexInfo    / list(shelf, book, page)
        -> when using a function with custom parameters
            Then the specialType variable should be set to "Model"
            - Model                  / list(function(wav, params), params)
        -> when using two functions with custom parameters for permittivity + permeability
            Then the specialType variable should be set to "ModelMu"
            - ModelMu                / [list(function(wav, params), params), list(function(wav, params), params)]

        And these materials have to be processed through the Material constructor first
        before being fed to Structure as Material objects

    """

    def __init__(self, mat, specialType="Default", verbose=False):

        if issubclass(mat.__class__, Material):
            # Has already been processed by this constructor previously
            if verbose:
                print("Preprocessed material:", mat.__name__)

        if specialType == "Default":
            # The default behaviors listed in the docstring
            self.specialType = specialType
            if mat.__class__.__name__ == "function":
                # Is a custom function that only takes the wavelength as a parameter
                self.type = "CustomFunction"
                self.permittivity_function = mat
                self.name = "CustomFunction: " + mat.__name__
                if verbose:
                    print(
                        "Custom dispersive material. Epsilon=",
                        mat.__name__,
                        "(wavelength in nm)",
                    )

            elif not hasattr(mat, "__iter__"):
                # no func / not iterable --> single value, convert to complex by default
                self.type = "simple_perm"
                self.name = "SimplePermittivity:" + str(mat)
                self.permittivity = complex(mat)
                if verbose:
                    print("Simple, non dispersive: epsilon=", self.permittivity)

            elif (
                isinstance(mat, list)
                and (isinstance(mat[0], float) or isinstance(mat[0], complex))
                and (isinstance(mat[1], float) or isinstance(mat[1], complex))
            ):
                # magnetic == [complex, complex]
                # iterable: if list or similar --> magnetic
                self.type = "magnetic"
                self.permittivity = mat[0]
                self.permeability = mat[1]
                self.name = (
                    "MagneticPermittivity:" + str(mat[0]) + "Permability:" + str(mat[1])
                )
                if verbose:
                    print("Magnetic, non dispersive: epsilon=", mat[0], " mu=", mat[1])
                if len(mat) > 2:
                    print(
                        f"Warning: Magnetic material should have 2 values (epsilon / mu), but {len(mat)} were given."
                    )

            elif isinstance(mat, str):
                # iterable: string --> database material from file in shipped database
                import pkgutil

                f = pkgutil.get_data(__name__, "data/material_data.json")
                f_str = f.decode(encoding="utf8")
                database = json.loads(f_str)
                if mat in database:
                    material_data = database[mat]
                    model = material_data["model"]

                    if model == "ExpData":
                        # Experimnental data to be interpolated
                        self.type = "ExpData"
                        self.name = "ExpData: " + str(mat)

                        wl = np.array(material_data["wavelength_list"])
                        epsilon = np.array(material_data["permittivities"])
                        if "permittivities_imag" in material_data:
                            epsilon = epsilon + 1j * np.array(
                                material_data["permittivities_imag"]
                            )

                        self.wavelength_list = np.array(wl, dtype=float)
                        self.permittivities = np.array(epsilon, dtype=complex)

                    elif model == "BrendelBormann":
                        # Brendel & Bormann model with all necessary parameters
                        self.type = "BrendelBormann"
                        self.name = "BrendelBormann model: " + str(mat)
                        self.f0 = material_data["f0"]
                        self.Gamma0 = material_data["Gamma0"]
                        self.omega_p = material_data["omega_p"]
                        self.f = np.array(material_data["f"])
                        self.gamma = np.array(material_data["Gamma"])
                        self.omega = np.array(material_data["omega"])
                        self.sigma = np.array(material_data["sigma"])

                    else:
                        print(model, " not an existing model (yet).")
                        # sys.exit()

                    if verbose:
                        print("Database material:", self.name)
                else:
                    print(mat, "Unknown material in the database (for the moment)")
                    # print("Known materials:\n", existing_materials())
                    # sys.exit()

            else:
                print(
                    f"Warning: Given data is not in the right format for a 'Default' specialType. You should check the data format or specify a specialType. You can refer to the following table:"
                )
                print(self.__doc__)

        elif specialType == "File":
            # Importing from file
            self.type = "ExpData"
            self.name = "ExpData: " + str(mat)

            file = mat
            data = np.loadtxt(file, dtype=float)

            wl = data[:, 0]
            n = data[:, 1]
            k = data[:, 2]

            self.wavelength_list = np.array(wl, dtype=float)
            self.permittivities = np.array((n + 1.0j * k) ** 2, dtype=complex)

        elif specialType == "RII":
            # Refractive index material
            if len(mat) != 3:
                print(
                    f"Warning: Material RefractiveIndex Database is expected to be a list of 3 values, but {len(mat)} were given."
                )
            self.type = "RefractiveIndexInfo"
            self.specialType = specialType
            self.name = "MaterialRefractiveIndexDatabase: " + str(mat)
            shelf, book, page = mat[0], mat[1], mat[2]
            material = RefractiveIndexMaterial(shelf, book, page)  # create object
            self.material = material
            if verbose:
                print("Material from Refractiveindex Database")
            if len(mat) != 3:
                print(
                    f"Warning: Material from RefractiveIndex Database should have 3 values (shelf, book, page), but {len(mat)} were given."
                )

        elif specialType == "Model":
            # A custom function that takes more parameters than simply the wavelength
            self.type = "Model"
            self.specialType = specialType
            self.permittivity_function = mat[0]
            self.params = [mat[i + 1] for i in range(len(mat) - 1)]
            self.name = "Customfunction: " + str(mat[0])

        elif specialType == "ModelMu":
            # Two custom functions that take more parameters than simply the wavelength
            self.type = "ModelMu"
            self.specialType = specialType
            eps = mat[0]
            self.permittivity_function = eps[0]
            self.eps_params = [eps[i + 1] for i in range(len(eps) - 1)]
            mu = mat[1]
            self.permeability_function = mu[0]
            self.mu_params = [mu[i + 1] for i in range(len(mu) - 1)]
            self.name = (
                "CustomfunctionMu: "
                + str(self.permittivity_function)
                + " "
                + str(self.permeability_function)
            )
            if verbose:
                print(
                    "CustomfunctionMu: "
                    + str(self.permittivity_function.__name__)
                    + " "
                    + str(self.permeability_function.__name__)
                )

        else:
            print(f"Warning: Unknown type : {specialType}")

    def __str__(self):
        return self.name

    def get_permittivity(self, wavelength):
        if self.type == "simple_perm":
            return self.permittivity

        elif self.type == "magnetic":
            return self.permittivity

        elif self.type == "CustomFunction":
            return self.permittivity_function(wavelength)

        elif self.type == "Model":
            return self.permittivity_function(wavelength, *self.params)

        elif self.type == "ModelMu":
            return self.permittivity_function(wavelength, *self.eps_params)

        elif self.type == "BrendelBormann":
            w = 6.62607015e-25 * 299792458 / 1.602176634e-19 / wavelength
            chi_b = 0
            for i in range(len(self.f)):
                a = np.sqrt(w * (w + 1j * self.gamma[i]))
                x = (a - self.omega[i]) / (np.sqrt(2) * self.sigma[i])
                y = (a + self.omega[i]) / (np.sqrt(2) * self.sigma[i])
                # Polarizability due to bound electrons
                erx = wofz(x)
                ery = wofz(y)
                oscill_strength = (
                    1j
                    * np.sqrt(np.pi)
                    * self.f[i]
                    * self.omega_p**2
                    / (2 * np.sqrt(2) * a * self.sigma[i])
                )
                chi_b += oscill_strength * (erx + ery)
            # Equivalent polarizability linked to free electrons (Drude model)
            chi_f = -self.omega_p**2 * self.f0 / (w * (w + 1j * self.Gamma0))
            epsilon = 1 + chi_f + chi_b
            return epsilon

        elif self.type == "ExpData":
            return np.interp(wavelength, self.wavelength_list, self.permittivities)

        elif self.type == "RefractiveIndexInfo":
            try:
                k = self.material.get_extinction_coefficient(wavelength)
                return self.material.get_epsilon(wavelength)
            except:
                n = self.material.get_refractive_index(wavelength)
                return n**2

    def get_permeability(self, wavelength, verbose=False):
        if self.type == "magnetic":
            return self.permeability
        if self.type == "ModelMu":
            return self.permeability_function(wavelength, *self.mu_params)
        elif self.type == "RefractiveIndexInfo":
            if verbose:
                print(
                    "Warning: Magnetic parameters from RefractiveIndex Database are not implemented. Default permeability is set to 1.0 ."
                )
            return 1.0
        return 1.0
