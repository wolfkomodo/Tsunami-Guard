import random
import pprint
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy

stress = 0

class Particle():
# I added the tagged attribute for debugging purposes
	def __init__(self, particletype, vy, vz, energy, x, y, z, tagged):
		self.particletype = particletype
		self.vy = vy
		self.vz = vz
		self.energy = energy
		self.x = x
		self.y = y
		self.z = z
		self.tagged = tagged
# finds energy value for different particle types
def energyval(flag):
	if flag == "w":
		return random.randint(1, 100)
	elif flag == "r":
		return -1
	else:
		return 0

# MATRIX CREATING FUNCTIONS

def find_wallmat(yval, xval, zval, val):
	if 0.75 * val < zval <= 0.95 * val and yval < 0.75 * val:
		return 'r'
	elif zval < 0.35 * val:
		return 'w'
	return 'a'

def pos_wall_matrix(dim):
	d = dim**3
	matrix = [[[Particle(find_wallmat(x, z, y, dim), int(round(random.uniform(-100, -50))), int(round(random.uniform(350, 600)))\
		, energyval(find_wallmat(x, z, y, dim)), z, y, x, False) for y in range(dim)]for x in range(dim)] for z in range(dim)]
	matrix[40][5][30].tagged = True
	return matrix

def find_gridmat(yval, xval, zval, val):
	if val * 0.6 < zval <= val * 0.89 and yval <= val * 0.96:
		if val * 0.15 < yval <= val * 0.36:
			if val * 0.1 < xval <= val * 0.3:
				return 'a'
			elif val * 0.4 < xval <= val * 0.6:
				return 'a'
			elif val * 0.7 < xval <= val * 0.9:
				return 'a'
		if val * 0.39 < yval <= val * 0.6:
			if val * 0.1 < xval <= val * 0.3:
				return 'a'
			elif val * 0.4 < xval <= val * 0.6:
				return 'a'
			elif val * 0.7 < xval <= val * 0.9:
				return 'a'
		if val * 0.63 < yval <= val * 0.84:
			if val * 0.1 < xval <= val * 0.3:
				return 'a'
			elif val * 0.4 < xval <= val * 0.6:
				return 'a'
			elif val * 0.7 < xval <= val * 0.9:
				return 'a'
		return 'r'
	elif zval < val * 0.35:
		return 'w'
	return 'a'

def pos_grid_matrix(dim):
	d = dim**3
	matrix = [[[Particle(find_gridmat(x, z, y, dim), int(round(random.uniform(-100, -50))), int(round(random.uniform(350, 600)))\
		, energyval(find_gridmat(x, z, y, dim)), z, y, x, False) for y in range(dim)]for x in range(dim)] for z in range(dim)]
	matrix[40][5][30].tagged = True
	return matrix

def find_xmat(xval, zval, yval):
	if 13 < xval <= 78:
		if 65 < zval <= 75 and yval <= 60:
			return 'r'
		elif 45 < zval <= 95 and 20 < yval <= 40:
			return 'r'
	if zval < 35:
		return 'w'
	return 'a'

def pos_x_matrix(dim):
	d = dim**3
	matrix = [[[Particle(find_xmat(z, y, x), int(round(random.uniform(-100, -50))), int(round(random.uniform(350, 600))), \
		energyval(find_xmat(z, y, x)), z, y, x, False) for y in range(dim)]for x in range(dim)] for z in range(dim)]
	matrix[40][5][30].tagged = True
	return matrix

# JAX ONLY WORKS ON 1000 x 1000 x 1000 grid

def find_simplejaxmat(yval, xval, zval):
	if 40 < xval <= 60 and 75 < zval<= 95 and yval <= 5:
		return 'r'
	elif 25 < xval <= 75 and 80 < zval <= 90 and 32 < yval <= 42:
		return 'r'
	elif 45 < xval <= 55 and 50 < zval <= 100 and 32 < yval <= 42:
		return 'r'
	elif xval <= 25 and 66 < zval <= 100 and 22 < yval <= 52:
		return 'r'
	elif 75 < xval <= 100 and 66 < zval <= 100 and 22 < yval <= 52:
		return 'r'
	elif 31 < xval < 69 and 35 < zval <= 60 and 22 < yval <= 52:
		return 'r'
	# elif 31 < xval < 69 and 100 < zval <= 125 and 22 < yval <= 52:
	# 	return 'r'
	if zval < 35:
		return 'w'
	return 'a'

def pos_jax_matrix(dim):
	d = dim**3
	matrix = [[[Particle(find_simplejaxmat(x, z, y), int(round(random.uniform(-100, -50))), int(round(random.uniform(350, 600)))\
		, energyval(find_simplejaxmat(x, z, y)), z, y, x, False) for y in range(dim)]for x in range(dim)] for z in range(dim)]
	matrix[40][5][30].tagged = True
	return matrix

# TETRAMAT HAS ONLY RIGHT ANGLES

def find_simpletetramat(yval, xval, zval):
	if 38 < xval <= 63 and 73 < zval <= 98 and yval <= 25:
		return 'r'
	elif 38 < xval <= 63 and 73 < zval <= 98 and yval <= 99:
		return 'r'
	elif 38 < xval <= 63 and 35 < zval <= 98 and yval <= 25:
		return 'r'
	elif 0 < xval <= 100 and 73 < zval <= 98 and yval <= 25:
		return 'r'
	elif zval < 35:
		return 'w'
	return 'a'

def pos_tetra_matrix(dim):
	d = dim**3
	matrix = [[[Particle(find_simpletetramat(x, z, y), int(round(random.uniform(-100, -50))), int(round(random.uniform\
		(350, 600))), energyval(find_simpletetramat(x, z, y)), z, y, x, False) for y in range(dim)]for x in range(dim)] for z in range(dim)]
	matrix[40][5][30].tagged = True
	return matrix

def find_hmat(xval, zval, yval):
	if 65 < zval <= 85:
		if yval <= 95:
			if xval > 80 or xval <= 20:
				return 'r'
			elif 40 < yval <= 60:
				return 'r'
	elif zval < 35:
		return 'w'
	return 'a'

def pos_h_matrix(dim):
	d = dim**3
	matrix = [[[Particle(find_hmat(z, y, x), int(round(random.uniform(-100, -50))), int(round(random.uniform\
		(350, 600))), energyval(find_hmat(z, y, x)), z, y, x, False) for y in range(dim)]for x in range(dim)] for z in range(dim)]
	matrix[40][5][30].tagged = True
	return matrix

def find_slitmat(xval, zval, yval):
	if 60 < zval <= 90:
		if 40 < xval < 60 and yval <= 10:
			return 'a'
		elif yval <= 52:
			return 'r'
	elif zval < 35:
		return 'w'
	return 'a'

def pos_slit_matrix(dim):
	d = dim**3
	matrix = [[[Particle(find_slitmat(z, y, x), int(round(random.uniform(-100, -50))), int(round(random.uniform\
		(350, 600))), energyval(find_slitmat(z, y, x)), z, y, x, False) for y in range(dim)]for x in range(dim)] for z in range(dim)]
	matrix[40][5][30].tagged = True
	return matrix

def find_trapmat(xval, zval, yval):
	if 75 < zval <= 95:
		if yval <= 50:
			return 'r'
	if 55 < zval <= 95:
		if zval >= yval:
			return 'r'
	elif zval < 35:
		return 'w'
	return 'a'

def pos_trap_matrix(dim):
	d = dim**3
	matrix = [[[Particle(find_trapmat(z, y, x), int(round(random.uniform(-100, -50))), int(round(random.uniform\
		(350, 600))), energyval(find_trapmat(z, y, x)), z, y, x, False) for y in range(dim)]for x in range(dim)] for z in range(dim)]
	matrix[40][5][30].tagged = True
	return matrix

def find_pointmat(yval, xval, zval):
	if 55 < zval <= 75:
		if  yval <= 3.75 * zval - 206.25:
			return 'r'
	if 75 < zval <= 95:
		if yval <= -3.75 * zval + 356.25:
			return 'r'
	elif zval < 35:
		return 'w'
	return 'a'

def pos_point_matrix(dim):
	d = dim**3
	matrix = [[[Particle(find_pointmat(x, z, y), int(round(random.uniform(-100, -50))), int(round\
		(random.uniform(350, 600))), energyval(find_pointmat(x, z, y)), z, y, x, False) for y in range(dim)]for x in range(dim)] for z in range(dim)]
	matrix[40][5][30].tagged = True
	return matrix

def find_ditchmat(yval, xval, zval):
	if 55 < zval <= 75:
		if  yval <= -3.75 * zval + 281.25:
			return 'r'
	if 75 < zval <= 95:
		if yval <= 3.75 * zval - 281.25:
			return 'r'
	elif zval < 35:
		return 'w'
	return 'a'

def pos_ditch_matrix(dim):
	d = dim**3
	matrix = [[[Particle(find_ditchmat(x, z, y), int(round(random.uniform(-100, -50))), int(round\
		(random.uniform(350, 600))), energyval(find_ditchmat(x, z, y)), z, y, x, False) for y in range(dim)]for x in range(dim)] for z in range(dim)]
	matrix[40][5][30].tagged = True
	return matrix

def find_trimat(yval, xval, zval):
	if 50 < zval <= 100:
		if yval <= zval * -1.2 + 120:
			return 'r'
		else:
			return 'a'
	elif zval < 35:
		return 'w'
	return 'a'

def pos_tri_matrix(dim):
	d = dim**3
	matrix = [[[Particle(find_trimat(x, z, y), int(round(random.uniform(-100, -50))), int(round\
		(random.uniform(350, 600))), energyval(find_trimat(x, z, y)), z, y, x, False) for y in range(dim)]for x in range(dim)] for z in range(dim)]
	matrix[40][5][30].tagged = True
	return matrix

def find_floatmat(yval, xval, zval, val):
	if val * 0.6 < zval <= val * 0.89 and val * 0.13 < yval <= val * 0.96:
		if val * 0.15 < yval <= val * 0.36:
			if val * 0.1 < xval <= val * 0.3:
				return 'a'
			elif val * 0.4 < xval <= val * 0.6:
				return 'a'
			elif val * 0.7 < xval <= val * 0.9:
				return 'a'
		if val * 0.39 < yval <= val * 0.6:
			if val * 0.1 < xval <= val * 0.3:
				return 'a'
			elif val * 0.4 < xval <= val * 0.6:
				return 'a'
			elif val * 0.7 < xval <= val * 0.9:
				return 'a'
		if val * 0.63 < yval <= val * 0.84:
			if val * 0.1 < xval <= val * 0.3:
				return 'a'
			elif val * 0.4 < xval <= val * 0.6:
				return 'a'
			elif val * 0.7 < xval <= val * 0.9:
				return 'a'
		return 'r'
	elif zval < val * 0.35:
		return 'w'
	return 'a'
def pos_float_matrix(dim):
	d = dim**3
	matrix = [[[Particle(find_floatmat(x, z, y, dim), int(round(random.uniform(-100, -50))), int(round(random.uniform(350, 600)))\
		, energyval(find_floatmat(x, z, y, dim)), z, y, x, False) for y in range(dim)]for x in range(dim)] for z in range(dim)]
	matrix[40][5][30].tagged = True
	return matrix

def find_xmat(xval, zval, yval):
	if 13 < xval <= 78:
		if 65 < zval <= 75 and yval <= 60:
			return 'r'
		elif 45 < zval <= 95 and 20 < yval <= 40:
			return 'r'
	if zval < 35:
		return 'w'
	return 'a'

############################################################################################################################################

# Finds total energy of barrier
def sum_energy(pos_matrix):
	total_energy = 0
	for x in pos_matrix:
		for y in x:
			for z in y:
				if z:
					if z.particletype == 'r':
						total_energy += z.energy
	return total_energy

# Calculates stress to be added to the barrier

def add_stress(particle):
	global stress
	return 0.025 * (((0.1 * (particle.y) * (math.sqrt((particle.vy**2) + (particle.vz**2))))/0.0063) * math.sin(math.atan(particle.vz/particle.vy)))
	
# Modified formula for floating barrier

def add_stress_mod(particle):
	global stress
	return 0.0025 * math.sqrt(particle.vy**2 + particle.vz**2)/0.0063


def test_mat(pos_matrix):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection = '3d')
	dim = len(pos_matrix)
	# lst_color = []
	xlist = []
	ylist = []
	zlist = []
	for x in range(dim):
		for y in range(dim):
			for z in range(dim):
				if pos_matrix[x][y][z].particletype == 'r':
						# lst_color.append('brown')
						xlist.append(x)
						ylist.append(z)
						zlist.append(y)
				# elif pos_matrix[x][y][z].particletype == 'w':
				# 	lst_color.append('blue')
				# 	xlist.append(x)
				# 	ylist.append(y)
				# 	zlist.append(z)
	ax.set_xlabel('X')
	ax.set_ylabel('Z')
	ax.set_zlabel('Y')
	ax.set_xlim3d(0, 100);
	ax.set_ylim3d(0, 100);
	ax.set_zlim3d(0, 100);
	ax.scatter(xlist, ylist, zlist, c='brown', marker='o')
	plt.show()


# Notes:
# Don't repeat yourself (DRY coding)
# Name them explicitly (don't use names like `d`)
# Variable naming format: "`datastructure_name`" (for example `coorlist` would be `lst_coords`)


def energy_homogeneity(position_matrix):
	homogeneity = 0
	lst_waters = get_type_coords(position_matrix, 'w')
	for water in lst_waters:
		neighbors = get_neighbors(position_matrix, water[0], water[1], water[2])
		lst_energy = [position_matrix[neighbor[0]][neighbor[1]][neighbor[2]].energy for neighbor in neighbors] + [position_matrix[water[0]][water[1]][water[2]].energy]
		homogeneity += np.std(lst_energy)
	return homogeneity


# Steps of Testing
# Generate position and energy matrix of specified dimensions
# Get the average energy, store it aside
# Use raise_energy(200)
# Get average energy, store it aside
# How many timesteps does this method take to let energy_homogeneity to be about 0?
# How much time does propagate energy take per iteration?
# Plot plot_homogeneity_propagation until about 0
# Plot energy_plot3D until about 0 (UNLESS the answer to number #5 is greater than 10)

def find_channels(matrix):
	lenm = len(matrix)
	channel = True
	list_tup = []
	for x in range(lenm):
		for z in range(lenm):
			for y in range(lenm):
				if matrix[x][y][z].particletype == "w" and channel:
					channel = True
				else:
					channel = False
		if channel:
			list_tup.append(('y', x, z))
				
	for y in range(lenm):
		for z in range(lenm):
			for x in range(lenm):
				if matrix[x][y][z].particletype == "w" and channel:
					channel = True
				else:
					channel = False
		if channel:
			list_tup.append(('x', y, z))
	return list_tup

def get_type_coords(position_matrix, flag):
	dim = len(position_matrix)
	lst = []
	if flag == 1:
		flag = "w"
	if flag == 0:
		flag = "r"
	if flag == -1:
		flag = "a"
	for x in range(dim):
		for y in range(dim):
			for z in range(dim):
				if position_matrix[x][y][z]:
					if position_matrix[x][y][z].particletype == flag:
						lst.append([x, y, z])
	return lst

def mutate(pos_matrix, energy_matrix):
	waters = get_type_coords(pos_matrix, 1)
	neighbors = []
	choice = []
	dim = len(pos_matrix)
	swap = 0
	for w in waters:
		if random.randint(1, 10) == 1:
			neighbors = get_neighbors(energy_matrix, w[0], w[1], w[2])
			if neighbors:
				choice = random.choice(neighbors)
				swap = energy_matrix[w[0]][w[1]][w[2]]
				energy_matrix[w[0]][w[1]][w[2]] = energy_matrix[choice[0]][choice[1]][choice[2]]
				energy_matrix[choice[0]][choice[1]][choice[2]] = swap
	return energy_matrix

def get_neighbors(matrix, x, y, z):
	coorlist = []
	dim = len(matrix)
	if x + 1 <= dim - 1:
		if matrix[x + 1][y][z]:
			if matrix[x + 1][y][z].particletype == "w":
				coorlist.append([x + 1, y, z])
	if x - 1 >= 0:
		if matrix[x - 1][y][z]:
			if matrix[x - 1][y][z].particletype == "w":
				coorlist.append([x - 1, y, z])
	if y + 1 <= dim - 1:
		if matrix[x][y + 1][z]:
			if matrix[x][y + 1][z].particletype == "w":
				coorlist.append([x, y + 1, z])
	if y - 1 >= 0:
		if matrix[x][y - 1][z]:
			if matrix[x][y - 1][z].particletype == "w":
				coorlist.append([x, y - 1, z])
	if z + 1 <= dim - 1:
		if matrix[x][y][z + 1]:
			if matrix[x][y][z + 1].particletype == "w":
				coorlist.append([x, y, z + 1])
	if z - 1 >= 0:
		if matrix[x][y][z - 1]:
			if matrix[x][y][z - 1].particletype == "w":
				coorlist.append([x, y, z - 1])
	return coorlist

def energy_matrix(matrix):
	energy = []
	dim = len(matrix)
	for x in range(dim):
		energy.append([])
		for y in range(dim):
			energy[x].append([])
			for z in range(dim):
				if matrix[x][y][z].particletype == "w":
					energy[x][y].append(random.randint(1, 100))
				elif matrix[x][y][z].particletype == "r":
					energy[x][y].append(-1)
				else:
					energy[x][y].append(0)
	return energy

def pos_small_matrix(dim):
	matrix = [[[Particle(find_wallmat(x, z, y, dim), int(round(random.uniform(1, 2))), int(round(random.uniform(1, 2))), energyval(find_wallmat(y, x, z)), z, y, x, False) for y in range(dim)]for x in range(dim)] for z in range(dim)]
	return matrix

# Evaluates and simulates collision of all kinds

def collide(xhit, yhit, zhit, x, y, z, position_matrix):
    global stress
    
    p_init = position_matrix[x][y][z]
    dim = len(position_matrix)
    if position_matrix:
            if xhit >= dim or zhit < 0 or yhit >= dim or xhit < 0 or zhit >= dim or zhit < 0:
                position_matrix[x][y][z] = None
            elif yhit < 0:
                p_init.vy *= -1
            elif position_matrix[xhit][yhit][zhit]:
                p_hit = position_matrix[xhit][yhit][zhit]
                if p_hit.particletype == 'w':
                    position_matrix = move_particle(xhit, yhit, zhit, position_matrix)
                    if p_hit:
                        p_init.__dict__, p_hit.__dict__ = \
                        p_hit.__dict__, p_init.__dict__
                    else:
                        p_hit.__dict__ = p_init.__dict__
                        position_matrix[x][y][z] = None
                elif p_hit.particletype == 'r':
                    movey = p_init.vy
                    movez = p_init.vz
                    p_hit.energy += 0.025 * (0.95 * math.sqrt(movey**2 + movez**2))
                    stress += add_stress_mod(p_init)
                    position_matrix[x][y][z] = None
                elif p_hit.particletype == 'a':
                    p_init.__dict__, p_hit.__dict__ = \
                    p_hit.__dict__, p_init.__dict__
            else:
                position_matrix[xhit][yhit][zhit] = position_matrix[x][y][z]
                position_matrix[x][y][z] = None
    return position_matrix

# Calculates new location for particles

def get_new_yz(x, y, z, vy, vz, t=0.0063):
    newy = int(y + vy*t - 193*t**2)
    newz = int(z + vz*t)
    return newy, newz

# Runs the collide function; used for recursive call

def move_particle(x, y, z, position_matrix):
    p_init = position_matrix[x][y][z]
    if position_matrix:
        if p_init and p_init.particletype == 'w':
            newy, newz = get_new_yz(x, y, z, p_init.vy, p_init.vz)
#             print(z, newz)
            if p_init.vy > 0:
                return collide(x, newy, newz, x, y, z, position_matrix)
            elif p_init.vy < 0:
                return collide(x, -1*newy, newz, x, y, z, position_matrix)

#  Performs movement at each timestep

def move_recompute(part_matrix):
    global printed
    dim = len(part_matrix)
    check_matrix = [[[True for x in range(dim)]for y in range(dim)]for z in range(dim)]
    for x in range(dim):
        for y in range(dim):
            for z in range(dim):
                if move_particle(x, y, z, part_matrix) != None:
                    if part_matrix[x][y][z]:
                        if part_matrix[x][y][z].particletype == 'w':
                            if check_matrix[x][y][z]:
                                part_matrix = move_particle(x, y, z, part_matrix)
                                check_matrix[x][y][z] = False
    return part_matrix

def compute_energies(pmatrix):
    # Iterate through the position matrix
    # Overwrite every energy value for waters with (vz^2 + vy^2)/(40400)
    for x in pmatrix:
        for y in x:
            for z in y:
                if z:
                    if z.particletype == 'w':
                        z.energy = (z.vz^2 + z.vy^2)/(40400)
    return pmatrix

hlist = []


# Plots the simulation

def plot(num_timesteps, pos_matrix):
    global hlist
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    dim = len(pos_matrix)
    lst_color = []
    xlist = []
    ylist = []
    zlist = []
    numthing = 0
    for i in range(num_timesteps):
        pos_matrix = move_recompute(pos_matrix)
        pos_matrix = compute_energies(pos_matrix)
        hlist.append(energy_homogeneity(pos_matrix))
        for x in pos_matrix:
            for y in x:
                for z in y:
                    if z:
                        if z.particletype == 'w':
                            numthing += 1
        print(numthing)
        numthing = 0
    for x in range(dim):
        for y in range(dim):
            for z in range(dim):
                if pos_matrix[x][y][z]:
                    if pos_matrix[x][y][z].particletype == 'r':
                        lst_color.append('brown')
                        xlist.append(x)
                        ylist.append(z)
                        zlist.append(y)
                    elif pos_matrix[x][y][z].particletype == 'w':
                        lst_color.append('blue')
                        xlist.append(x)
                        ylist.append(z)
                        zlist.append(y)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.set_xlim3d(0, dim);
    ax.set_ylim3d(0, dim);
    ax.set_zlim3d(0, dim);
    ax.scatter(xlist, ylist, zlist, c=lst_color, marker='o')
    plt.show()

# Shows distribution of energy in barrier

def energy_wall3D(position_matrix):
    global x_listing
    global y_listing
    global z_listing
    global nrg_listing
    dim = len(position_matrix)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    xlist = []
    ylist = []
    zlist = []
    lst_nrg = []
    for x in range(dim):
        for y in range(dim):
            for z in range(dim):
                if position_matrix[x][y][z]:
                    if position_matrix[x][y][z].particletype == 'r' and position_matrix[x][y][z].energy != -1:
                        lst_nrg.append(position_matrix[x][y][z].energy)
    for x in range(dim):
        for y in range(dim):
            for z in range(dim):
                if position_matrix[x][y][z]:
                    if position_matrix[x][y][z].particletype == 'r' and position_matrix[x][y][z].energy != -1:

                        xlist.append(x)
                        ylist.append(z)
                        zlist.append(y)
    np.asarray(lst_nrg)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.set_title('Energy Graph')
    ax.set_xlim3d(0, 100);
    ax.set_ylim3d(0, 100);
    ax.set_zlim3d(0, 100);
    for x in lst_nrg:
        x += 1000
    x_listing = xlist
    y_listing = ylist
    z_listing = zlist
    nrg_listing = lst_nrg
    ax.scatter(xlist, ylist, zlist, c = lst_nrg, marker='o', s = 2)
    plt.show()









