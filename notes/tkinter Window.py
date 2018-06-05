import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import pandas as pd
from tkinter import filedialog, Tk, Frame, Label, Button
#import tkinter as tk
from PIL import Image, ImageTk
import math
import random
import pprint
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df_new = []
pos_matrix = []
matrix = []
class mainclass(Tk):
    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)
        Tk.wm_title(self, "Wave Energetics")

        container = Frame(self, background="red")
        container.pack(side="top", fill="both", expand = True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.geometry("666x800")
        self.frames = {}
        for F in (StartPage, PageOne, SimPage):

            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

class StartPage(Frame):

    def __init__(self, parent, controller):
        Frame.__init__(self, parent, background="mintcream")
        titleLabel = Label(self, text="Wave Breaker Genesis", fg="DodgerBlue2", bg="mintcream", font=("Times", 20))
        titleLabel.pack(side="top")
        nameLabel = Label(self, text="Akshay Attaluri", fg="royalblue1", bg="mintcream", font=("Times", 17))
        nameLabel.pack()
        button1 = Button(self, text="Click to Continue", bg="mintcream",
                            command=lambda: controller.show_frame(PageOne))
        button1.pack(side="bottom")

        # harvard = Image(self, file="/Users/rattaluri001/Desktop/wave-energetics/Photos/Harvard.png")
        # harvard.place(x=0, y=0, anchor="SE")
        # python = Image(self, file="/Users/rattaluri001/Desktop/wave-energetics/Photos/Python.png")
        # python.place(x=20, y=0, anchor="SE")

        # TO COMPLETE TOMORROW
        # harvard_img = Image.open('/Users/rattaluri001/Desktop/wave-energetics/Photos/Harvard.png').resize((30, 33), Image.ANTIALIAS)
        # harvard = ImageTk.PhotoImage(harvard_img)
        # harvard_lbl = Label(self, image=harvard, height=33, width=30, bg="mintcream")
        # harvard_lbl.image = harvard
        # harvard_lbl.pack(side="right", anchor="se")

        python_img = Image.open('/Users/rattaluri001/Desktop/wave-energetics/Photos/Python.png').resize((30, 30))
        python = ImageTk.PhotoImage(python_img)
        python_lbl = Label(self, image=python, height=30, width=30, bg="mintcream")
        python_lbl.image = python
        python_lbl.pack(side="right", anchor="se")
class PageOne(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent, background="mintcream")
        titleLabel = Label(self, text="Wave Breaker Genesis", fg="DodgerBlue2", bg="mintcream", font=("Times", 20))
        titleLabel.pack(side="top")
        spaceLabel11 = Label(self, text="", bg="mintcream")
        spaceLabel11.pack()
        buttonn = Button(self, text='Choose File',
                            command=self.askopenfile)
        buttonn.pack()
        button2 = Button(self, text="Back to Home",
                              command = lambda: controller.show_frame(StartPage))

        button2.pack(side="bottom")
        button2 = Button(self, text="Continue",
                         command=lambda: controller.show_frame(SimPage))

        button2.pack(side="bottom")

    def askopenfile(self):
        global df_new
        global pos_matrix
        global matrix
        df_name = filedialog.askopenfilename(filetypes=[("Shape Files", "*.csv")])
        df_new = pd.read_csv(df_name)
        #edgelen = len(df_new[0].unique())
        edgelen = 100
        pos_matrix = [[[df_new["VALUE"].iloc[edgelen ** 2 * x + edgelen * y + z]
                           for z in range(edgelen)]
                          for y in range(edgelen)]
                         for x in range(edgelen)]
        for x in range(100):
            for y in range(100):
                for z in range(100):
                    if pos_matrix[x][y][z] == 1:
                        pos_matrix[x][y][z] = 'w'
                    elif pos_matrix[x][y][z] == 0:
                        pos_matrix[x][y][z] = 'r'
                    else:
                        pos_matrix[x][y][z] = 'a'
        matrix = [[[Particle(pos_matrix[x][y][z], int(round(random.uniform(5, 10))),
                             int(round(random.uniform(5, 10))), self.energyval(pos_matrix[x][y][z]), z, y, x, False)
                    for y in range(edgelen)] for x in range(edgelen)] for z in range(edgelen)]

    def energyval(self, flag):
        if flag == "w":
            return random.randint(1, 100)
        elif flag == "r":
            return -1
        else:
            return 0

        # tuplist = []
        # for x in range(len(output_matrix)):
        #     for y in range(len(output_matrix[x])):
        #         for z in range(len(output_matrix[x][y])):
        #             tuplist.append((x, y, z, output_matrix[x][y][z]))
        # df_matrix = pd.DataFrame(tuplist)
        # df_matrix.columns = ["X", "Y", "Z", "VALUE"]
        # df_matrix.to_csv("/Users/rattaluri001/Desktop/wave-energetics/input_files/test_file.csv", index=False)


class SimPage(Frame):

    def __init__(self, parent, controller):
        global pos_matrix
        global matrix
        Frame.__init__(self, parent, background="mintcream")
        titleLabel = Label(self, text="Tsunami Guard", fg="DodgerBlue2", bg="mintcream", font=("Times", 20))
        titleLabel.pack(side="top")
        button1 = Button(self, text="Back", bg="mintcream",
                            command=lambda: controller.show_frame(PageOne))
        button1.pack(side="bottom")
        button2 = Button(self, text="Graph", bg="mintcream",
                        command=lambda: self.plot(matrix))
        button2.pack(side="bottom")
        ################################################################################################################
        # FIGURE
    def plot(self, pos_matrix):
        fig = Figure(figsize=(5, 5), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        dim = len(pos_matrix)
        print(dim)
        lst_color = []
        xlist = []
        ylist = []
        zlist = []
        for n in range(0):
            pos_matrix = self.move_recompute(pos_matrix)
        for x in range(dim):
            for y in range(dim):
                for z in range(dim):
                    if pos_matrix[x][y][z]:
                        if pos_matrix[x][y][z].particletype == 'r':
                            lst_color.append("brown")
                            xlist.append(x)
                            ylist.append(y)
                            zlist.append(z)
                        elif pos_matrix[x][y][z].particletype == 'w':
                            lst_color.append("blue")
                            xlist.append(x)
                            ylist.append(y)
                            zlist.append(z)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim3d(0, 100);
        ax.set_ylim3d(0, 100);
        ax.set_zlim3d(0, 100);
        ax.scatter(xlist, ylist, zlist, c=lst_color, marker='o')

        canvas = FigureCanvasTkAgg(fig, self)
        canvas.show()
        canvas.get_tk_widget().pack(side="bottom", fill="both", expand=True)
    def collide(self, xhit, yhit, zhit, x, y, z, position_matrix):
        if position_matrix:
            if position_matrix[x][y][z]:
                if xhit > 99 or yhit > 99 or zhit > 99 or xhit < 0 or zhit < 0:
                    position_matrix[x][y][z] = None
                elif yhit < 0:
                    position_matrix[x][y][z].vy *= -1
                elif position_matrix[xhit][yhit][zhit]:
                    if position_matrix[xhit][yhit][zhit].particletype == 'w':
                        position_matrix = self.move_particle(xhit, yhit, zhit, position_matrix)
                        if position_matrix[xhit][yhit][zhit]:
                            position_matrix[x][y][z].__dict__, position_matrix[xhit][yhit][zhit].__dict__ = \
                                position_matrix[xhit][yhit][zhit].__dict__, position_matrix[x][y][z].__dict__
                        else:
                            position_matrix[xhit][yhit][zhit] = position_matrix[x][y][z]
                            position_matrix[x][y][z] = None
                    elif position_matrix[xhit][yhit][zhit].particletype == 'r':
                        movey = position_matrix[x][y][z].vy
                        movez = position_matrix[x][y][z].vz
                        position_matrix[x][y][z].energy += 0.95 * math.sqrt(movey ** 2 + movez ** 2)
                        position_matrix[x][y][z] = None
                    elif position_matrix[xhit][yhit][zhit].particletype == 'a':
                        position_matrix[x][y][z].__dict__, position_matrix[xhit][yhit][zhit].__dict__ = \
                            position_matrix[xhit][yhit][zhit].__dict__, position_matrix[x][y][z].__dict__
                else:
                    position_matrix[xhit][yhit][zhit] = position_matrix[x][y][z]
                    position_matrix[x][y][z] = None
        return position_matrix
    def move_particle(self, x, y, z, position_matrix):
        if position_matrix:
            if position_matrix[x][y][z] and position_matrix[x][y][z].particletype == 'w':
                movey = position_matrix[x][y][z].vy
                movez = position_matrix[x][y][z].vz
                if movey > 0:
                    return self.collide(x, int(round(
                        math.tan(math.atan(movey / movez)) * (z + movez) - (4.9 / movez ** 2) * (z + movez) ** 2)),
                                   z + movez, x, y, z, position_matrix)
                elif movey < 0:
                    return self.collide(x, -1 * int(round(
                        math.tan(math.atan(movey / movez)) * (z + movez) - (4.9 / movez ** 2) * (z + movez) ** 2)),
                                   z + movez, x, y, z, position_matrix)
    def move_recompute(self, part_matrix):
        check_matrix = [[[True for x in range(100)] for y in range(100)] for z in range(100)]
        for x in range(100):
            for y in range(100):
                for z in range(100):
                    if self.move_particle(x, y, z, part_matrix) != None:
                        if part_matrix[x][y][z]:
                            if part_matrix[x][y][z].particletype == 'w':
                                if check_matrix[x][y][z]:
                                    part_matrix = self.move_particle(x, y, z, part_matrix)
                                    check_matrix[x][y][z] = False
        return part_matrix
class Particle():
# I added the tagged attribute for debugging purposes
    def __init__(self, particletype, vy, vz, energy, x, y, z):
        self.particletype = particletype
        self.vy = vy
        self.vz = vz
        self.energy = energy
        self.x = x
        self.y = y
        self.z = z

app = mainclass()
app.mainloop()