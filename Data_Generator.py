import netgen.gui
from ngsolve import *
from ngsolve import meshes as me
from netgen.geom2d import *
from ngsolve import *
from ngsolve import internal as ngsint
import os
import PIL
import numpy as np
import openpyxl as xl
import pandas as pd
import pickle
import time

from netgen.geom2d import SplineGeometry


def create_data(uin, vmax, display_figure, save_images, save_outputs):
    nu = 0.001  # viscosity

    tau = 0.0001  # step size
    tend = 5  # time length of sim

    # Visualization Options
    from ngsolve import internal as ngsint
    # ngsint.viewoptions.drawoutline=0
    # ngsint.visoptions.gridsize = 75
    # ngsint.visoptions.showsurfacesolution = 1
    # ngsint.viewoptions.drawoutline=0
    # ngsint.viewoptions.drawcolorbar=0
    ngsint.visoptions.lineartexture = 1

    # -----Variable Parameters END----------

    # Newly defined geometry parameters
    geo_length = 2
    geo_height = 0.41
    num_vertx = 256
    num_verty = 66
    geo_vertx_spacing = geo_length / (num_vertx - 1)
    geo_verty_spacing = geo_height / (num_verty - 1)
    side = 0.1  # 2*0.05
    geo = SplineGeometry()

    geo.AddRectangle((0, 0), (geo_length, geo_height), bcs=("wall", "outlet", "wall", "inlet"))
    # geo.AddCircle((geoHeight / 2, geoHeight / 2), r=0.05, leftdomain=0, rightdomain=1, bc="cyl", maxh=0.02)
    geo.AddRectangle((math.floor((geo_height / 2 - side / 2) / geo_vertx_spacing) * geo_vertx_spacing,
                      math.floor((geo_height / 2 - side / 2) / geo_verty_spacing) * geo_verty_spacing), (
                         math.floor((geo_height / 2 + side / 2) / geo_vertx_spacing) * geo_vertx_spacing,
                         math.floor((geo_height / 2 + side / 2) / geo_verty_spacing) * geo_verty_spacing),
                     leftdomain=0,
                     rightdomain=1, bc="rectangle", maxh=0.02)

    mesh = Mesh(geo.GenerateMesh(maxh=0.07))


    Draw(mesh)
    mesh.Curve(3)

    # V = VectorH1(mesh, order=3, dirichlet="wall|cyl|inlet")
    V = VectorH1(mesh, order=3, dirichlet="wall|rectangle|inlet")

    Q = H1(mesh, order=3)

    X = FESpace([V, Q])

    u, p = X.TrialFunction()
    v, q = X.TestFunction()

    stokes = nu * InnerProduct(grad(u), grad(v)) + div(u) * q + div(v) * p - 1e-10 * p * q
    a = BilinearForm(X)
    a += stokes * dx
    a.Assemble()

    # nothing here ...
    f = LinearForm(X)
    f.Assemble()

    # gridfunction for the solution
    gfu = GridFunction(X)

    # parabolic inflow at inlet:
    gfu.components[0].Set(uin, definedon=mesh.Boundaries("inlet"))

    # solve Stokes problem for initial conditions:
    inv_stokes = a.mat.Inverse(X.FreeDofs())

    res = f.vec.CreateVector()
    res.data = f.vec - a.mat * gfu.vec
    gfu.vec.data += inv_stokes * res

    # matrix for implicit Euler
    mstar = BilinearForm(X)
    mstar += SymbolicBFI(u * v + tau * stokes)
    mstar.Assemble()
    inv = mstar.mat.Inverse(X.FreeDofs(), inverse="sparsecholesky")

    # the non-linear term
    conv = BilinearForm(X, nonassemble=True)  # convective term
    conv += (grad(u) * u) * v * dx

    # configuration for visualization
    Draw(Norm(gfu.components[0]), mesh, "velocity", sd=3, min=0, max=vmax * 1.2, autoscale=False)  # velocity magnitude
    # Draw (gfu.components[1], mesh, "Pressure", sd=3)  # Pressure
    # Draw (gfu.components[0][0], mesh, "velocity_x", sd=3)  #X component of velocity
    # Draw (gfu.components[0][1], mesh, "velocity_y", sd=3)  #Y component of velocity

    # implicit Euler/explicit Euler splitting method:
    # Initializes numpy arrays for pressure and velocity
    pressure_data = np.ones((num_verty, num_vertx))
    velocity_data_x = np.ones((num_verty, num_vertx))
    velocity_data_y = np.ones((num_verty, num_vertx))
    flow_tensor = np.ones((3, num_verty, num_vertx))

    out_of_bound_counter = 0
    for z in range(num_vertx * num_verty):
        row = int(z / num_vertx)
        column = (z % num_vertx)
        if not (mesh.Contains(geo_vertx_spacing * column, geo_verty_spacing * row)):
            out_of_bound_counter = out_of_bound_counter + 1
    x_points = np.ones(num_verty * num_vertx - out_of_bound_counter)
    y_points = np.ones(num_verty * num_vertx - out_of_bound_counter)
    points_contained = np.zeros((num_verty, num_vertx))
    temp_counter = 0
    for z in range(num_vertx * num_verty):
        row = int(z / num_vertx)
        column = (z % num_vertx)
        if mesh.Contains(geo_vertx_spacing * column, geo_verty_spacing * row):
            x_points[temp_counter] = geo_vertx_spacing * column
            y_points[temp_counter] = geo_verty_spacing * row
            points_contained[row, column] = 1
            temp_counter += 1
        else:
            points_contained[row, column] = 0

    i = 0
    t = 0
    file_data_counter = 0  # counts the number of exported CSVs
    png_count = 0  # counts the amount of exported images

    # initialize variable to configure the frequency at which the solution is checked for steady state.
    i_ss_check = int(1 / tau)
    if i_ss_check < 1:
        print("ERROR: SS check less than 1")
        exit()

    def make_vtk():
        vtk = VTKOutput(ma=mesh,
                        coefs=[gfu.components[0]],
                        names=["velocity"],
                        filename=vtk_filename,
                        subdivision=3)
        return vtk

    with TaskManager():
        while t < tend:
            print(vmax, " t=", t, )

            conv.Apply(gfu.vec, res)
            res.data += a.mat * gfu.vec
            gfu.vec.data -= tau * inv * res

            if display_figure is True and i % 200 == 0:
                Redraw()
                if save_images:
                    netgen.gui.Snapshot(w=2000, h=2000,
                                        filename="./PNGs/" + str(vmax) + "-" + "{0:0=6d}".format(png_count) + ".png")
                    png_count = png_count + 1

            if i % i_ss_check == 0 and False:
                if i == 0:
                    old_pressure_data_flat = gfu.components[1](mesh(x_points, y_points))
                    old_velocity_data_x_flat = gfu.components[0][0](mesh(x_points, y_points))
                    old_velocity_data_y_flat = gfu.components[0][1](mesh(x_points, y_points))
                else:
                    pressure_data_flat = gfu.components[1](mesh(x_points, y_points))
                    velocity_data_x_flat = gfu.components[0][0](mesh(x_points, y_points))
                    velocity_data_y_flat = gfu.components[0][1](mesh(x_points, y_points))
                    pressure_half = (np.amax(np.absolute(pressure_data_flat)) + np.amin(
                        np.absolute(pressure_data_flat))) / 2
                    velocity_x_half = (np.amax(np.absolute(velocity_data_x_flat)) + np.amin(
                        np.absolute(velocity_data_x_flat))) / 2
                    velocity_y_half = (np.amax(np.absolute(velocity_data_y_flat)) + np.amin(
                        np.absolute(velocity_data_y_flat))) / 2
                    relative_diff_pressure = np.sum(
                        ((pressure_data_flat - old_pressure_data_flat) ** 2) / (pressure_half ** 2)) / len(
                        pressure_data_flat)
                    relative_diff_velocity_x = np.sum(
                        ((velocity_data_x_flat - old_velocity_data_x_flat) ** 2) / (velocity_x_half ** 2)) / len(
                        velocity_data_x_flat)
                    relative_diff_velocity_y = np.sum(
                        ((velocity_data_y_flat - old_velocity_data_y_flat) ** 2) / (velocity_y_half ** 2)) / len(
                        velocity_data_y_flat)
                    print("t=", t, )
                    print("Relative difference in pressure: " + str(relative_diff_pressure))
                    print("Relative difference in v_x: " + str(relative_diff_velocity_x))
                    print("Relative difference in v_y: " + str(relative_diff_velocity_y))
                    old_pressure_data_flat = pressure_data_flat
                    old_velocity_data_x_flat = velocity_data_x_flat
                    old_velocity_data_y_flat = velocity_data_y_flat
                    if relative_diff_pressure + relative_diff_velocity_x + relative_diff_velocity_y < 1e-3:
                        # TEMPORARILY DISABLED due to false positives caused by low velocities
                        break

            if i % 200 == 0 and save_outputs is True:

                # NOTE ABOUT ELEMENTS (50x50 cells):10201 total, 2601 (51^2) vertices,2500 (50^2) elements
                # ALSO midpoint of lines. 50*51,51*50 sum of all is 10201 = gridfunc
                pressure_data_flat = gfu.components[1](mesh(x_points, y_points))
                velocity_data_x_flat = gfu.components[0][0](mesh(x_points, y_points))
                velocity_data_y_flat = gfu.components[0][1](mesh(x_points, y_points))
                temp_counter = 0
                for z in range(num_vertx * num_verty):
                    row = int(z / num_vertx)
                    column = (z % num_vertx)
                    if points_contained[row, column] == 1:
                        pressure_data[row, column] = pressure_data_flat[temp_counter]
                        velocity_data_x[row, column] = velocity_data_x_flat[temp_counter]
                        velocity_data_y[row, column] = velocity_data_y_flat[temp_counter]
                        temp_counter += 1
                    else:
                        pressure_data[row, column] = 0
                        velocity_data_x[row, column] = 0
                        velocity_data_y[row, column] = 0

                wb = xl.Workbook()
                wb.create_sheet("Data Sheet")
                ws = wb.active
                filepath_p = "{0:0=6d}".format(file_data_counter) + " t; " + str(t) + " v; " + str(
                    vmax) + " Pressure.csv"
                filepath_vx = "{0:0=6d}".format(file_data_counter) + " t; " + str(t) + " v; " + str(
                    vmax) + " Velocity X.csv"
                filepath_vy = "{0:0=6d}".format(file_data_counter) + " t; " + str(t) + " v; " + str(
                    vmax) + " Velocity Y.csv"
                # filepathP2 = "stew interpolated P.csv"
                df_p = pd.DataFrame(pressure_data)
                df_vx = pd.DataFrame(velocity_data_x)
                df_vy = pd.DataFrame(velocity_data_y)
                # df2 = pd.DataFrame(stewP2)
                df_p.to_csv("./CSVs/" + str(vmax) + "/" + filepath_p, index=False)
                df_vx.to_csv("./CSVs/" + str(vmax) + "/" + filepath_vx, index=False)
                df_vy.to_csv("./CSVs/" + str(vmax) + "/" + filepath_vy, index=False)
                file_data_counter = file_data_counter + 1

            i = i + 1

            t = t + tau

            #     vtk = make_vtk()
            #     vtk.Do()

# --- CREATES A LIST OF VELOCITIES TO RUN
# count = 0.1
# upper = 1
# increment = 0.025
# velocity_vector = count
# while count < upper:
#     count = count + increment
#     velocity_vector = np.append(velocity_vector, count)
# print(velocity_vector)

vtk_filename = "vtk_output_1"

# ----LOOPABLE VERSION

velocity_vector = (0.1125, 0.3125, 0.5125)
for vmax in velocity_vector:
    print(vmax)
    # os.mkdir("./CSVs/val" + str(vmax) + "/") # makes file folders if none are there
    uin = CoefficientFunction((vmax * 4 * y * (0.41 - y) / (0.41 * 0.41), 0))
    create_data(uin, vmax, display_figure=False, save_images=False, save_outputs=True)

exit()

# ----ONE SHOT VERSION
vmax = 0.001
vtk_filename = "./LIC/Circle_inv vmax0.001 t10"
uin = CoefficientFunction((vmax * 4 * y * (0.41 - y) / (0.41 * 0.41), 0))
create_data(uin, vmax, display_figure=False, save_images=False, save_outputs=False)
