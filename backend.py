

import csv
import tkinter
from tkinter import messagebox as msb
from tkinter import scrolledtext
from tkinter.filedialog import *
import scipy.stats as stats
import pandas as pd
import numpy as np
import math as mat
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from string import ascii_letters
from functools import partial
from string import ascii_letters
from pylab import title, figure, xlabel, ylabel, xticks, bar, legend, axis, savefig
from fpdf import FPDF


matplotlib.use('TkAgg')

#

# lista, all_header, selected_data, selected_header

#
#  ------------- KONTROLA DANYCH --------------
#

#


def has_header(file, nrows=20):
    df = pd.read_csv(file, header=None, nrows=nrows,
                     sep=None, delimiter=None,
                     engine='python', encoding='iso-8859-2')
    df_header = pd.read_csv(file, nrows=nrows,
                            sep=None, delimiter=None,
                            engine='python', encoding='iso-8859-2')

    if (tuple(df.dtypes) != tuple(df_header.dtypes)) == True:
        return pd.read_csv(file, header=0,
                           sep=None, delimiter=None,
                           engine='python', encoding='iso-8859-2')
    else:
        return pd.read_csv(file, header=None,
                           sep=None, delimiter=None,
                           engine='python', encoding='iso-8859-2')


def wczytaj_plik(scierzeka1, scierzeka2, przycisk_1, przycisk_2, przycisk_3):
    global lista, all_header
    path = tkinter.filedialog.askopenfilename(
        initialdir='/',
        title='Wybierz plik csv',
        filetypes=[('text files', '.csv')])
    lista = has_header(path)
    all_header = list(lista.columns)

    scierzeka1.config(text=path)
    scierzeka2.config(text=path)
    odpal(przycisk_1)
    odpal(przycisk_2)
    odpal(przycisk_3)


class WybieranieKolumn(tkinter.Tk):
    def __init__(self , *args, **kwargs):
        tkinter.Tk.__init__(self, *args, **kwargs)
        self.resizable(width=False, height=False)
        self.title('Wybierz kolumny')

        # ustawienie

        frame4 = Frame(self)
        frame5 = Frame(self)
        frame6 = Frame(self)
        frame7 = Frame(self)
        frame8 = Frame(self)
        frame9 = Frame(self)
        frame10 = Frame(self)
        frame4.grid(row=1, column=0, rowspan=5, sticky=W + E)
        frame5.grid(row=1, column=2, rowspan=5, sticky=W + E)
        frame6.grid(row=6, column=2, sticky=W + E)
        frame7.grid(row=1, column=1, sticky=W + E)
        frame8.grid(row=2, column=1, sticky=W + E)
        frame9.grid(row=4, column=1, sticky=W + E)
        frame10.grid(row=5, column=1, sticky=W + E)

        #   lista z tego pobieram

        self.Rama1 = tkinter.Frame(frame4)
        self.Pasek = tkinter.Scrollbar(self.Rama1)
        self.ListaBox = tkinter.Listbox(self.Rama1)
        self.ListaBox.delete(0, tkinter.END)
        for i in range(len(all_header)):
            self.ListaBox.insert(tkinter.END, all_header[i])
        self.Pasek['command'] = self.ListaBox.yview
        self.ListaBox['yscrollcommand'] = self.Pasek.set
        self.ListaBox.bind('<<ListboxSelect>>')

        #   Lista do tego pobieram i przyciski

        self.Rama2 = tkinter.Frame(frame5)
        self.Pasek2 = tkinter.Scrollbar(self.Rama2)
        self.ListaBox2 = tkinter.Listbox(self.Rama2)
        self.Pasek2['command'] = self.ListaBox2.yview
        self.ListaBox2['yscrollcommand'] = self.Pasek2.set
        self.ListaBox2.bind('<<ListboxSelect>>')

        self.B_wybor = tkinter.Button(frame6, text='OK', command=self.wyjmij_z_listy)
        self.B_jedna_plus = tkinter.Button(frame7, text='>', command=self.daj_jedno)
        self.B_wszystkie_plus = tkinter.Button(frame8, text='>>', command=self.dodaj_wszystki)
        self.B_jedna_minus = tkinter.Button(frame9, text='< ', command=self.usun_jedno)
        self.B_wszystkie_minus = tkinter.Button(frame10, text='<< ', command=self.usun_wszystki)

        # packi

        self.Rama1.pack()
        self.Pasek.pack(side=tkinter.RIGHT, fill=tkinter.Y)
        self.ListaBox.pack(side=tkinter.LEFT, fill=tkinter.Y)

        self.Rama2.pack()
        self.Pasek2.pack(side=tkinter.RIGHT, fill=tkinter.Y)
        self.ListaBox2.pack(side=tkinter.LEFT, fill=tkinter.Y)

        self.B_wybor.pack()
        self.B_jedna_plus.pack()
        self.B_wszystkie_plus.pack()
        self.B_jedna_minus.pack()
        self.B_wszystkie_minus.pack()

    def daj_jedno(self):
        a = str((self.ListaBox.get(self.ListaBox.curselection())))
        self.ListaBox2.insert(tkinter.END, a)

    def dodaj_wszystki(self):
        self.ListaBox2.delete(0, tkinter.END)
        for i in range(len(all_header)):
            self.ListaBox2.insert(tkinter.END, all_header[i])

    def usun_jedno(self):
        self.ListaBox2.delete(tkinter.ANCHOR)

    def usun_wszystki(self):
        self.ListaBox2.delete(0, tkinter.END)

    def wyjmij_z_listy(self):
        global selected_header
        selected_header = list(self.ListaBox2.get(0, tkinter.END))
        # Aktywoj.config(state="normal")
        self.destroy()


def wybierz_kolumny():
    klasa_kolumny = WybieranieKolumn()
    klasa_kolumny.mainloop()


def zatwierdz_kolumny(przycisk_1, przycisk_2, przycisk_3, przycisk_4, przycisk_6, przycisk_7):
    global selected_data
    selected_data = lista[selected_header]

    odpal(przycisk_1)
    odpal(przycisk_2)
    odpal(przycisk_3)
    odpal(przycisk_4)
    odpal(przycisk_6)
    odpal(przycisk_7)


def wybierz_do_wypisania(dane, okno, ile_wierszy):
    x = 0
    for index, value in dane.items():
        wez = tkinter.StringVar(okno)
        pokaz = tkinter.Label(okno, textvariable=wez)
        wez.set(index)
        pokaz.grid(row=0, column=x)
        x += 1

    for x in range(dane.shape[0]):
        for y in range(ile_wierszy):
            wez = tkinter.StringVar(okno)
            pokaz = tkinter.Label(okno, textvariable=wez)
            wez.set(dane.iloc[y, x])
            pokaz.grid(row=y + 1, column=x)


class PodgladWybrane(tkinter.Tk):
    def __init__(self, *args, **kwargs):
        tkinter.Tk.__init__(self, *args, **kwargs)
        self.resizable(width=False, height=False)

        if selected_data.shape[0] > 20:
            wybierz_do_wypisania(selected_data, self, 20)
        else:
            wybierz_do_wypisania(selected_data, self, selected_data.shape[0])


def podglad_kolumn():
    klasa_wyniki = PodgladWybrane()
    klasa_wyniki.mainloop()


class PodgladWszystko(tkinter.Tk):
    def __init__(self, *args, **kwargs):
        tkinter.Tk.__init__(self, *args, **kwargs)
        self.resizable(width=False, height=False)

        if lista.shape[0] > 20:
            wybierz_do_wypisania(lista, self, 20)
        else:
            wybierz_do_wypisania(lista, self, lista.shape[0])


def podglad_wszystko():
    wyswietl_wszystko = PodgladWszystko()
    wyswietl_wszystko.mainloop()


def zapis_pliku():

    save = pd.DataFrame(data=selected_data)

    files = [('csv', '*.csv')]
    file_name = asksaveasfilename(filetypes=files, defaultextension=files)

    if file_name:
        save.to_csv(file_name, sep=';', encoding='utf-8-sig')


def notatnik():
    okno = Tk()
    okno.title("Notatki")
    okno.geometry('350x810')
    notatki = scrolledtext.ScrolledText(okno, width=40, height=50)
    notatki.grid(column=0, row=0)
    okno.mainloop()


#


#

"""
#  ------------- WSPEIRAJĄCE --------------
"""

#


#


def odpal(tabela):
    tabela.config(state="normal")


def utworz_dataframe(dane, index, columny):
    np_w = np.array(dane)
    return pd.DataFrame(data=np_w, index=index, columns=columny)


def tworzenie_tabel_w_petli(dane, okno, poziom):
    if poziom == 'True':
        for x in range(len(dane)):
            wez = tkinter.StringVar(okno)
            pokaz = tkinter.Label(okno, textvariable=wez)
            a = dane[x]
            wez.set(a)
            pokaz.grid(row=1, column=x + 2)
    else:
        for y in range(len(dane)):
            wez = tkinter.StringVar(okno)
            pokaz = tkinter.Label(okno, textvariable=wez)
            a = dane[y]
            wez.set(a)
            pokaz.grid(row=y + 2, column=1)


def wypelanianie_tabeli_w_petli(dlugosc, okno, x):
    for y in range(dlugosc):
        okno.b2 = tkinter.Text(okno, width=10, height=1)
        okno.b2.insert('end', okno.Wyniki[y])
        okno.b2.config(state="disabled")
        okno.b2.grid(row=y + 2, column=x + 2)


def donothig():
    pass


def donothing():
    pass


#


#

"""
#                       Miary Położenia
"""

#


#


class MiaryPol(tkinter.Tk):
    def __init__(self, *args, **kwargs):
        tkinter.Tk.__init__(self, *args, **kwargs)
        self.resizable(width=False, height=False)

        self.funkcje = [
            'Średnia arytmetyczna',
            'Średnia harmoniczna',
            'Wartość minimalna',
            'Kwartyl dolny',
            'Mediana',
            'Kwartyl górny',
            'Wartość maksymalna']
        self.save = []

        for x in range(len(selected_header)):
            self.Wyniki = []
            describe = selected_data.iloc[:, x].describe()
            sr_h = stats.hmean(selected_data.iloc[:, x], axis=0, dtype=None)

            self.Wyniki.append(describe[1])
            self.Wyniki.append(sr_h)
            for i in range(3, 8):
                self.Wyniki.append(describe[i])

            self.save.append(self.Wyniki)

            wypelanianie_tabeli_w_petli(len(self.funkcje), self, x)

        tworzenie_tabel_w_petli(selected_header, self, poziom='True')

        tworzenie_tabel_w_petli(self.funkcje, self, poziom='False')

        self.l1 = Button(self, text='Zapisz wyniki', command=self.zapisz)
        self.l1.grid(row=len(self.funkcje) + 3, column=len(selected_header) + 1, pady=10, sticky=W)

        self.wolny = Label(self, text=' ', padx=10, pady=10)
        self.wolny.grid(row=len(self.funkcje) + 3, column=len(selected_header) + 3)

    def zapisz(self):

        files = [('csv', '*.csv')]
        file_name = asksaveasfilename(filetypes=files, defaultextension=files)

        if file_name:
            utworz_dataframe(self.save, selected_header, self.funkcje).to_csv(file_name, sep=';', encoding='utf-8-sig')


def miary_polozenia():
    klasa_wyniki = MiaryPol()
    klasa_wyniki.mainloop()


class MiaryZmi(tkinter.Tk):
    def __init__(self, *args, **kwargs):
        tkinter.Tk.__init__(self, *args, **kwargs)
        self.resizable(width=False, height=False)

        self.funkcje = (
            'Wariancja',
            'Odchylenie standardowe',
            'Odchylenie przeciętne',
            'Klasyczny współczynnik zmienności',
            'Rozstęp',
            'Rozstęp międzykwartylowy',
            'Odchylenie ćwiartkowe',
            'Pozycyjny współczynnik zmienności')

        self.save = []

        for x in range(len(selected_header)):
            self.Wyniki = []

            war = float(np.nanvar(selected_data.iloc[:, x]))
            odch_s = np.nanstd(selected_data.iloc[:, x1])
            suma = 0
            for i in range(selected_data.shape[0]):
                suma = suma + mat.fabs(selected_data.iloc[i, x] - np.nanmean(selected_data.iloc[:, x]))
            odch_p = suma / selected_data.shape[0]
            klas_wsp_z = (np.nanstd(selected_data.iloc[:, x]) / np.nanmean(selected_data.iloc[:, x])) * 100
            roz = np.ptp(selected_data.iloc[:, x])
            roz_m = stats.iqr(selected_data.iloc[:, x])
            odch_c = stats.iqr(selected_data.iloc[:, x])/2
            poz_wsp_z = ((stats.iqr(selected_data.iloc[:, x]) / 2) / np.nanmedian(
                np.sort(selected_data.iloc[:, x]))) * 100

            self.Wyniki.append(war)
            self.Wyniki.append(odch_s)
            self.Wyniki.append(odch_p)
            self.Wyniki.append(klas_wsp_z)
            self.Wyniki.append(roz)
            self.Wyniki.append(roz_m)
            self.Wyniki.append(odch_c)
            self.Wyniki.append(poz_wsp_z)

            self.save.append(self.Wyniki)

            wypelanianie_tabeli_w_petli(len(self.funkcje), self, x)

        tworzenie_tabel_w_petli(selected_header, self, poziom='True')

        tworzenie_tabel_w_petli(self.funkcje, self, poziom='False')

        self.l1 = Button(self, text='Zapisz wyniki', command=self.zapisz)
        self.l1.grid(row=len(self.funkcje) + 3, column=len(selected_header) + 1, pady=10, sticky=W)

        self.wolny = Label(self, text=' ', padx=10, pady=10)
        self.wolny.grid(row=len(self.funkcje) + 3, column=len(selected_header) + 3)

    def zapisz(self):

        files = [('csv', '*.csv')]
        file_name = asksaveasfilename(filetypes=files, defaultextension=files)

        if file_name:
            utworz_dataframe(self.save, selected_header, self.funkcje).to_csv(file_name, sep=';', encoding='utf-8-sig')


def miary_zmiennosci():
    klasa_wyniki = MiaryZmi()
    klasa_wyniki.mainloop()


def kappa():
    pass


class MiaryAsy(tkinter.Tk):
    def __init__(self, *args, **kwargs):
        tkinter.Tk.__init__(self, *args, **kwargs)
        self.resizable(width=False, height=False)

        self.funkcje = ('Wskaźnik skośności',
                        'Pozycyjny wskaźnik skośności',
                        'Pozycyjny współczynnik asymetrii',
                        'Klasyczny współczynnik asymetrii',
                        'Współczynnik kurtozy',
                        'Współczynnik ekscesu')
        self.save = []

        for x in range(len(selected_header)):
            sko = stats.skew(selected_data.iloc[:, x])
            y = np.sort(selected_data.iloc[:, x])
            poz_sko = np.nanquantile(y, q=0.75) + np.nanquantile(y, q=0.25) - 2 * (np.nanmedian(y))
            poz_asy = poz_sko / (np.nanquantile(y, q=0.75) - np.nanquantile(y, q=0.25))
            mean = np.nanmean(selected_data.iloc[:, x])
            a = 0
            for i in range(selected_data.shape[0]):
                a = a + ((selected_data.iloc[i, x] - mean) ** 3)
                m3 = a / selected_data.shape[0]
            kla_asy = m3 / (np.nanstd(selected_data.iloc[:, x]) ** 3)
            kurtoza = stats.kurtosis(selected_data.iloc[:, x], axis=0, fisher=False)
            k1 = (stats.kurtosis(selected_data.iloc[:, x], axis=0, fisher=False)) - 3

            self.Wyniki = []

            self.Wyniki.append(sko)
            self.Wyniki.append(poz_sko)
            self.Wyniki.append(poz_asy)
            self.Wyniki.append(kla_asy)
            self.Wyniki.append(kurtoza)
            self.Wyniki.append(k1)

            self.save.append(self.Wyniki)

            wypelanianie_tabeli_w_petli(len(self.funkcje), self, x)

        tworzenie_tabel_w_petli(selected_header, self, poziom='True')

        tworzenie_tabel_w_petli(self.funkcje, self, poziom='False')

        self.l1 = Button(self, text='Zapisz wyniki', command=self.zapisz)
        self.l1.grid(row=len(self.funkcje) + 3, column=len(selected_header) + 1, pady=10, sticky=W)

        self.wolny = Label(self, text=' ', padx=10, pady=10)
        self.wolny.grid(row=len(self.funkcje) + 3, column=len(selected_header) + 3)

    def zapisz(self):

        files = [('csv', '*.csv')]
        file_name = asksaveasfilename(filetypes=files, defaultextension=files)


        if file_name:
            utworz_dataframe(self.save, selected_header, self.funkcje).to_csv(file_name, sep=';', encoding='utf-8-sig')


def miary_asymetrii():
    klasa_wyniki = MiaryAsy()
    klasa_wyniki.mainloop()


def corelation():
    corr = selected_data.corr()


    okno = Toplevel()

    okno.title("Macierz korelacji Pearsona")

    for x in range(len(list(corr.columns))):

        for y in range(len(list(corr.columns))):
            okno.b2 = tkinter.Text(okno, width=10, height=1)
            okno.b2.insert('end', corr.iloc[x, y])
            okno.b2.config(state="disabled")
            okno.b2.grid(row=y + 2, column=x + 2)

    tworzenie_tabel_w_petli(list(corr.columns), okno, poziom='True')
    tworzenie_tabel_w_petli(list(corr.columns), okno, poziom='False')


def kow():
    corr = selected_data.cov()

    okno = Toplevel()

    okno.title("Macierz Kowariancji")

    for x in range(len(list(corr.columns))):

        for y in range(len(list(corr.columns))):
            okno.b2 = tkinter.Text(okno, width=10, height=1)
            okno.b2.insert('end', corr.iloc[x, y])
            okno.b2.config(state="disabled")
            okno.b2.grid(row=y + 2, column=x + 2)

    tworzenie_tabel_w_petli(list(corr.columns), okno, poziom='True')
    tworzenie_tabel_w_petli(list(corr.columns), okno, poziom='False')


def kor_sper():

    corr = selected_data.corr(method='spearman')


    okno = Toplevel()

    okno.title("Macierz korelacji Spearmana")

    for x in range(len(list(corr.columns))):

        for y in range(len(list(corr.columns))):
            okno.b2 = tkinter.Text(okno, width=10, height=1)
            okno.b2.insert('end', corr.iloc[x, y])
            okno.b2.config(state="disabled")
            okno.b2.grid(row=y + 2, column=x + 2)

    tworzenie_tabel_w_petli(list(corr.columns), okno, poziom='True')
    tworzenie_tabel_w_petli(list(corr.columns), okno, poziom='False')


def reg_lin():
    pass


def reg_wyk():
    pass


def reg_kwadt():
    pass


#


#

"""
#                       Wykresy
"""

#


def zmien_text(box, zmienna, self):
    box.config(state=NORMAL)
    box.delete(0, END)
    box.insert(END, zmienna)
    box.config(state=DISABLED)
    self.destroy()


def wybor_zmiennej_do_wykresu(box, tekst):
    okno = tkinter.Toplevel()

    text = 'Która zmienna na osi ' + tekst + '?'

    wyswietl = Label(okno, text=text)
    wyswietl.pack()
    for i in range(len(selected_header)):
        buttonExample = tkinter.Button(okno, text=selected_header[i],
                                       command=partial(zmien_text, box, selected_header[i], okno))
        buttonExample.pack()


#

# selected_data, selected_header


def draw_graph(graph, x, y, e1, e2, e3):

    X = x.get()
    Y = y.get()

    e1 = e1.get()
    e2 = e2.get()
    e3 = e3.get()

    rysuj = tkinter.Toplevel()

    fig = graph(X, Y, e1, e2, e3)

    canvas = FigureCanvasTkAgg(fig, master=rysuj)  # A tk.DrawingArea.
    canvas.draw()
    canvas.get_tk_widget().pack()

    tkinter.mainloop()


def wykres_korelacji():

    rysuj = tkinter.Toplevel()

    corr = selected_data.corr()

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.set_title("Wykres koreacji", fontsize=16)

    sns.heatmap(data=corr, square=True, annot=True, cbar=True)

    canvas = FigureCanvasTkAgg(fig, master=rysuj)  # A tk.DrawingArea.
    canvas.draw()
    canvas.get_tk_widget().pack()

    tkinter.mainloop()


def wykres_relacji():

    sns.pairplot(data=selected_data)
    plt.show()


def wykres_1(x, y, nazwa_wykres, nazwa_x, nazwa_y):

    f, ax = plt.subplots(figsize=(10, 8))

    ax.set_title(nazwa_wykres, fontsize=16)
    ax.set_ylabel(nazwa_y, fontsize=14)
    ax.set_xlabel(nazwa_x, fontsize=14)

    sns.lineplot(x=selected_data[x], y=selected_data[y],  data=selected_data)

    return f


def wykres_2(x, y, nazwa_wykres, nazwa_x, nazwa_y):
    f, ax = plt.subplots(figsize=(10, 8))

    ax.set_title(nazwa_wykres, fontsize=16)
    ax.set_ylabel(nazwa_y, fontsize=14)
    ax.set_xlabel(nazwa_x, fontsize=14)

    sns.lmplot(data=selected_data, x=x, y=y)

    return f


def wykres_3(x, y, nazwa_wykres, nazwa_x, nazwa_y):
    f, ax = plt.subplots(figsize=(10, 8))

    ax.set_title(nazwa_wykres, fontsize=16)
    ax.set_ylabel(nazwa_y, fontsize=14)
    ax.set_xlabel(nazwa_x, fontsize=14)

    sns.distplot(selected_data[x])

    return f


def wykres_4(x, y, nazwa_wykres, nazwa_x, nazwa_y):

    f, ax = plt.subplots(figsize=(10, 8))

    ax.set_title(nazwa_wykres, fontsize=16)
    ax.set_ylabel(nazwa_y, fontsize=14)
    ax.set_xlabel(nazwa_x, fontsize=14)

    sns.distplot(selected_data[x], kde=True, hist = False)
    return f


def wykres_5(x, y, nazwa_wykres, nazwa_x, nazwa_y):
    f, ax = plt.subplots(figsize=(10, 8))

    ax.set_title(nazwa_wykres, fontsize=16)
    ax.set_ylabel(nazwa_y, fontsize=14)
    ax.set_xlabel(nazwa_x, fontsize=14)

    sns.residplot(data=selected_data, x=x, y=y, lowess=True)

    return f


def wykres_6(x, y, nazwa_wykres, nazwa_x, nazwa_y):

    f, ax = plt.subplots(figsize=(12, 8))

    ax.set_title(nazwa_wykres, fontsize=16)
    ax.set_ylabel(nazwa_y, fontsize=14)
    ax.set_xlabel(nazwa_x, fontsize=14)

    sns.barplot(x=x, y=y, data=selected_data, label="Total")

    return f


def wykres_7(x, y, nazwa_wykres, nazwa_x, nazwa_y):

    f, ax = plt.subplots(figsize=(12, 8))

    ax.set_title(nazwa_wykres, fontsize=16)
    ax.set_ylabel(nazwa_y, fontsize=14)
    ax.set_xlabel(nazwa_x, fontsize=14)

    sns.boxplot(x=selected_data[x], y=selected_data[y], data=selected_data)
    sns.despine(offset=10, trim=True)
    return f


def wykres_8(x, y, nazwa_wykres, nazwa_x, nazwa_y):
    f, ax = plt.subplots(figsize=(12, 8))

    ax.set_title(nazwa_wykres, fontsize=16)
    ax.set_ylabel(nazwa_y, fontsize=14)
    ax.set_xlabel(nazwa_x, fontsize=14)

    sns.violinplot(x=selected_data[x], y=selected_data[y])
    sns.despine(left=True)

    return f


def wykres_9(x, y, nazwa_wykres, nazwa_x, nazwa_y):
    f, ax = plt.subplots(figsize=(12, 8))

    ax.set_title(nazwa_wykres, fontsize=16)
    ax.set_ylabel(nazwa_y, fontsize=14)
    ax.set_xlabel(nazwa_x, fontsize=14)

    sns.boxenplot(data=selected_data, x=selected_data[x], y=selected_data[y], scale="linear")
    return f


def wykres_10(x, y, nazwa_wykres, nazwa_x, nazwa_y):

    k = selected_data[x]
    df = pd.DataFrame(data=k)
    a = df.groupby(k).count()

    f, ax = plt.subplots(figsize=(12, 8))
    ax.pie(a, labels=list(a.index), autopct='%1.1f%%', shadow=True, startangle=90)
    ax.axis('equal')

    """
    ax.set_title(nazwa_wykres, fontsize=16)
    ax.set_ylabel(nazwa_y, fontsize=14)
    ax.set_xlabel(nazwa_x, fontsize=14)
    """
    return f


def wykres_11(x, y, nazwa_wykres, nazwa_x, nazwa_y):

    f, ax = plt.subplots(figsize=(10, 8))

    ax = sns.barplot(data=selected_data, x=x, y=y)

    ax.set_title(nazwa_wykres, fontsize=16)
    ax.set_ylabel(nazwa_x, fontsize=14)
    ax.set_xlabel(nazwa_y, fontsize=14)

    return f


def wykres_12(x, y, nazwa_wykres, nazwa_x, nazwa_y):

    f, ax = plt.subplots(figsize=(10, 8))

    ax = sns.barplot(data=selected_data, x=x, y=y, orient='h')

    ax.set_title(nazwa_wykres, fontsize=16)
    ax.set_ylabel(nazwa_x, fontsize=14)
    ax.set_xlabel(nazwa_y, fontsize=14)

    return f


def wykres_13(x, y, nazwa_wykres, nazwa_x, nazwa_y):

    f, ax = plt.subplots(figsize=(10, 8))

    ax = sns.regplot(data=selected_data, x=x, y=y)

    ax.set_title(nazwa_wykres, fontsize=16)
    ax.set_ylabel(nazwa_x, fontsize=14)
    ax.set_xlabel(nazwa_y, fontsize=14)

    return f


def wykres_14(x, y, nazwa_wykres, nazwa_x, nazwa_y):
    f, ax = plt.subplots(figsize=(8, 10))
    ax = sns.barplot(data=selected_data, x=x, y=y)

    ax.set_title(nazwa_wykres, fontsize=16)
    ax.set_ylabel(nazwa_x, fontsize=14)
    ax.set_xlabel(nazwa_y, fontsize=14)

    return f


def wykres_15(x, y, nazwa_wykres, nazwa_x, nazwa_y):

    pass
#


# RAPORT


#


def stworz_raport(var1, var2, var3, var4): #, w_1, x11, y11, n11):

    a = var1.get()
    b = var2.get()
    c = var3.get()
    d = var4.get()

    #w1 = w_1.get()

    pdf = FPDF()
    pdf.add_page()
    pdf.set_xy(0, 0)
    pdf.set_font('arial', 'B', 12)
    pdf.cell(60)
    pdf.cell(75, 10, "Raport", 0, 2, 'C')
    pdf.cell(-37.5)

    if a == 1:
        m1 = selected_data.describe().T.round(decimals=2)

        pdf.set_font('arial', 'B', 12)
        pdf.cell(90, 10, " ", 0, 2, 'C')

        pdf.cell(14.7, 10, 'kolumna', 1, 0, 'C')
        pdf.cell(14.7, 10, 'count', 1, 0, 'C')
        pdf.cell(14.7, 10, 'mean', 1, 0, 'C')
        pdf.cell(14.7, 10, 'std', 1, 0, 'C')
        pdf.cell(14.7, 10, 'min', 1, 0, 'C')
        pdf.cell(14.7, 10, '25%', 1, 0, 'C')
        pdf.cell(14.7, 10, '50%', 1, 0, 'C')
        pdf.cell(14.7, 10, '75%', 1, 0, 'C')
        pdf.cell(14.7, 10, 'max', 1, 2, 'C')
        pdf.cell(-117.6)
        pdf.set_font('arial', '', 12)
        for i in range(0, len(m1)):
            pdf.cell(14.7, 10, '%s' % (selected_header[i]), 1, 0, 'C')
            pdf.cell(14.7, 10, '%s' % (str(m1['count'].iloc[i])), 1, 0, 'C')
            pdf.cell(14.7, 10, '%s' % (str(m1['mean'].iloc[i])), 1, 0, 'C')
            pdf.cell(14.7, 10, '%s' % (str(m1['std'].iloc[i])), 1, 0, 'C')
            pdf.cell(14.7, 10, '%s' % (str(m1['min'].iloc[i])), 1, 0, 'C')
            pdf.cell(14.7, 10, '%s' % (str(m1['25%'].iloc[i])), 1, 0, 'C')
            pdf.cell(14.7, 10, '%s' % (str(m1['50%'].iloc[i])), 1, 0, 'C')
            pdf.cell(14.7, 10, '%s' % (str(m1['75%'].iloc[i])), 1, 0, 'C')
            pdf.cell(14.7, 10, '%s' % (str(m1['max'].iloc[i])), 1, 2, 'C')
            pdf.cell(-117.6)

    if b == 1:
        m1 = miary_2()
        pdf.set_font('arial', 'B', 12)
        pdf.cell(90, 10, " ", 0, 2, 'C')

        pdf.cell(16.5, 10, 'kolumna', 1, 0, 'C')
        pdf.cell(16.5, 10, 'war', 1, 0, 'C')
        pdf.cell(16.5, 10, 'prze', 1, 0, 'C')
        pdf.cell(16.5, 10, 'kwz', 1, 0, 'C')
        pdf.cell(16.5, 10, 'roz', 1, 0, 'C')
        pdf.cell(16.5, 10, 'roz m', 1, 0, 'C')
        pdf.cell(16.5, 10, 'od c', 1, 0, 'C')
        pdf.cell(16.5, 10, 'pwz', 1, 2, 'C')

        pdf.cell(-115.5)
        pdf.set_font('arial', '', 12)

        for i in range(0, len(m1)):
            pdf.cell(16.5, 10, '%s' % (selected_header[i]), 1, 0, 'C')
            pdf.cell(16.5, 10, '%s' % (str(m1['Wariancja'].iloc[i])), 1, 0, 'C')
            pdf.cell(16.5, 10, '%s' % (str(m1['Odchylenie przeciętne'].iloc[i])), 1, 0, 'C')
            pdf.cell(16.5, 10, '%s' % (str(m1['Klasyczny współczynnik zmienności'].iloc[i])), 1, 0, 'C')
            pdf.cell(16.5, 10, '%s' % (str(m1['Rozstęp'].iloc[i])), 1, 0, 'C')
            pdf.cell(16.5, 10, '%s' % (str(m1['Rozstęp międzykwartylowy'].iloc[i])), 1, 0, 'C')
            pdf.cell(16.5, 10, '%s' % (str(m1['Odchylenie ćwiartkowe'].iloc[i])), 1, 0, 'C')
            pdf.cell(16.5, 10, '%s' % (str(m1['Pozycyjny współczynnik zmienności'].iloc[i])), 1, 2, 'C')

            pdf.cell(-115.5)

    if c == 1:
        m1 = miary_3()
        pdf.set_font('arial', 'B', 12)
        pdf.cell(90, 10, " ", 0, 2, 'C')

        pdf.cell(18.8, 10, 'kolumna', 1, 0, 'C')
        pdf.cell(18.8, 10, 'ws', 1, 0, 'C')
        pdf.cell(18.8, 10, 'pws', 1, 0, 'C')
        pdf.cell(18.8, 10, 'pwa', 1, 0, 'C')
        pdf.cell(18.8, 10, 'kwa', 1, 0, 'C')
        pdf.cell(18.8, 10, 'wk', 1, 0, 'C')
        pdf.cell(18.8, 10, 'we', 1, 2, 'C')

        pdf.cell(-112.8)
        pdf.set_font('arial', '', 12)

        for i in range(0, len(m1)):
            pdf.cell(18.8, 10, '%s' % (selected_header[i]), 1, 0, 'C')
            pdf.cell(18.8, 10, '%s' % (str(m1['Wskaźnik skośności'].iloc[i])), 1, 0, 'C')
            pdf.cell(18.8, 10, '%s' % (str(m1['Pozycyjny wskaźnik skośności'].iloc[i])), 1, 0, 'C')
            pdf.cell(18.8, 10, '%s' % (str(m1['Pozycyjny współczynnik asymetrii'].iloc[i])), 1, 0, 'C')
            pdf.cell(18.8, 10, '%s' % (str(m1['Klasyczny współczynnik asymetrii'].iloc[i])), 1, 0, 'C')
            pdf.cell(18.8, 10, '%s' % (str(m1['Współczynnik kurtozy'].iloc[i])), 1, 0, 'C')
            pdf.cell(18.8, 10, '%s' % (str(m1['Współczynnik ekscesu'].iloc[i])), 1, 2, 'C')

            pdf.cell(-112.8)

    if d == 1:
        cor_r = selected_data.corr()

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title("Wykres koreacji", fontsize=16)
        sns.heatmap(data=cor_r, square=True, annot=True, cbar=True)

        axis([0, 10, 0, 8])
        savefig('hatemap.png')
        pdf.image('hatemap.png', x=None, y=None, w=0, h=0, type='', link='')
        os.remove("hatemap.png")
    """
    if w1 == 1:
        x1 = x11.get()
        y1 = y11.get()
        n1 = n11.get()

        f, ax = plt.subplots(figsize=(10, 10))
        ax.set_title(n1, fontsize=16)

        sns.lineplot(data=selected_data, x=selected_data[x1], y=selected_data[y1])

        axis([0, 10, 0, 8])
        savefig('wykres1.png')
        pdf.image('wykres1.png', x=None, y=None, w=0, h=0, type='', link='')
        os.remove("wykres1.png")
    """
    pdf.cell(90, 10, " ", 0, 2, 'C')

    pdf.cell(90, 10, " ", 0, 2, 'C')

    pdf.output('Raport.pdf', 'F')


def miary_2():
    funkcje = (
        'Wariancja',
        'Odchylenie przeciętne',
        'Klasyczny współczynnik zmienności',
        'Rozstęp',
        'Rozstęp międzykwartylowy',
        'Odchylenie ćwiartkowe',
        'Pozycyjny współczynnik zmienności')

    save = []

    for x in range(len(selected_header)):
        Wyniki = []

        war = float(np.nanvar(selected_data.iloc[:, x]))
        suma = 0
        for i in range(selected_data.shape[0]):
            suma = suma + mat.fabs(selected_data.iloc[i, x] - np.nanmean(selected_data.iloc[:, x]))
        odch_p = suma / selected_data.shape[0]
        klas_wsp_z = (np.nanstd(selected_data.iloc[:, x]) / np.nanmean(selected_data.iloc[:, x])) * 100
        roz = np.ptp(selected_data.iloc[:, x])
        roz_m = stats.iqr(selected_data.iloc[:, x])
        odch_c = stats.iqr(selected_data.iloc[:, x])
        poz_wsp_z = ((stats.iqr(selected_data.iloc[:, x]) / 2) / np.nanmedian(
            np.sort(selected_data.iloc[:, x]))) * 100

        Wyniki.append(war)
        Wyniki.append(odch_p)
        Wyniki.append(klas_wsp_z)
        Wyniki.append(roz)
        Wyniki.append(roz_m)
        Wyniki.append(odch_c)
        Wyniki.append(poz_wsp_z)

        save.append(Wyniki)

    return pd.DataFrame(data=save, index=selected_header, columns=funkcje).round(decimals=2)


def miary_3():
    funkcje = ('Wskaźnik skośności',
               'Pozycyjny wskaźnik skośności',
               'Pozycyjny współczynnik asymetrii',
               'Klasyczny współczynnik asymetrii',
               'Współczynnik kurtozy',
               'Współczynnik ekscesu')
    save = []

    for x in range(len(selected_header)):
        sko = stats.skew(selected_data.iloc[:, x])
        y = np.sort(selected_data.iloc[:, x])
        poz_sko = np.nanquantile(y, q=0.75) + np.nanquantile(y, q=0.25) - 2 * (np.nanmedian(y))
        poz_asy = poz_sko / (np.nanquantile(y, q=0.75) - np.nanquantile(y, q=0.25))
        mean = np.nanmean(selected_data.iloc[:, x])
        a = 0
        for i in range(selected_data.shape[0]):
            a = a + ((selected_data.iloc[i, x] - mean) ** 3)
            m3 = a / selected_data.shape[0]
        kla_asy = m3 / (np.nanstd(selected_data.iloc[:, x]) ** 3)
        kurtoza = stats.kurtosis(selected_data.iloc[:, x], axis=0, fisher=False)
        k1 = (stats.kurtosis(selected_data.iloc[:, x], axis=0, fisher=False)) - 3

        Wyniki = []

        Wyniki.append(sko)
        Wyniki.append(poz_sko)
        Wyniki.append(poz_asy)
        Wyniki.append(kla_asy)
        Wyniki.append(kurtoza)
        Wyniki.append(k1)

        save.append(Wyniki)

    return pd.DataFrame(data=save, index=selected_header, columns=funkcje).round(decimals=2)










