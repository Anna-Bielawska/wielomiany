import numpy as np
import sys
import warnings

input_file = sys.argv[1]

with open(input_file, 'r') as f:
    lines = f.read()  
arr = [float(l) for l in lines.splitlines()]
wspolczynniki = arr[1:]
stopien = int(arr[0])
tekst = ''

for ind, wsp in enumerate(reversed(wspolczynniki)):
    ind = stopien - ind

    if wsp == int(wsp):
        wsp = int(wsp)

    if ind == 0:
        if wsp > 0:
            tekst += f' + {wsp}'
        if wsp < 0:
            tekst += f' {wsp}'
        continue
    if ind == 1:
        if wsp > 0:
            tekst += f' + {wsp}x'
        if wsp < 0:
            tekst += f' {wsp}x'
        continue
    if ind == stopien:
        if wsp > 0:
            if wsp == 1: 
                tekst += f'x^{stopien}'
            else:
                tekst += f'{wsp}x^{stopien}'
        if wsp < 0:
            if wsp == -1:
                tekst += f'-x^{stopien}'
            else:
                tekst += f'{wsp}x^{stopien}'
        continue

    if wsp > 0:
        if wsp == 1:
            tekst += f' + x^{ind}'
        else:
            tekst += f' + {wsp}x^{ind}'
    if wsp < 0:
        if wsp == -1:
            tekst += f' -x^{ind}'
        else:
            tekst += f' {wsp}x^{ind}'

if len(wspolczynniki) > stopien + 1:
    print("Wystąpił błąd w pliku, liczba współczynników wielomianu jest za duża.")
    exit()

if len(wspolczynniki) < stopien + 1:
    print("Wystąpił błąd w pliku, liczba współczynników wielomianu jest za mała.")
    exit()

A = -np.array(wspolczynniki[0:-1]).reshape((1,-1))/wspolczynniki[stopien] # potrzebujemy miec a_n = 1 dla wielomianu stopnia n

Matrix = np.zeros(shape=(stopien, stopien))

Matrix[stopien-1,::] = A # ostatni wiersz macierzy zastąp przeskalowanymi współczynnikami wielomianu, pomijając współczynnik a_n

for i in range(stopien-1):
    Matrix[i,i+1] = 1

srodki_kol = np.diag(Matrix)

maks_odl_srodkow = np.abs(min(srodki_kol)) + np.abs(max(srodki_kol))

R_lista = [np.sum(np.abs(Matrix), axis=1) - np.abs(srodki_kol)]

a0 = (min(srodki_kol) + max(srodki_kol)) / 2

R = maks_odl_srodkow/2 + np.max(R_lista) #znajdź dwa skrajne środki kół Gershgorina, obliczmy ich odległość od siebie, następnie dopasujmy promień okręgu o środku w tym punkcie
zero_vector = np.random.randn(1, 2).view(np.complex128) 

x0 = a0 + complex(zero_vector / np.abs(zero_vector) * R)

def newton(f, Df, x0, epsilon, n_steps=500, h=1e-6):

    xn_1 = x0 + 1
    xn = x0
    k = 1

    while( (k <= n_steps) and (np.abs(xn_1 - xn) > epsilon) ):
        xn_1 = xn
        fxn = f(xn)

        Dfxn = Df(xn)
        if np.abs(f(xn)) < 1e-16:
            break
        xn = xn - fxn/Dfxn
        k = k+1

    if k > n_steps:
        x0 = a0 + complex(zero_vector / np.abs(zero_vector) * R)
        newton(f,Df,x0,epsilon,n_steps)

    else:

        if len(pierwiastki) != 0:
            for z in pierwiastki.keys():
                if np.abs(xn - z) < 1e-8:
                    # print("Znaleziono pierwiastek wielokrotny.")
                    pierwiastki[z] += 1
                    break
                else: 
                    # print("Znaleziono nowy pierwiastek.")
                    pierwiastki[xn] = 1
                    break

        else:
            pierwiastki[xn] = 1

        if sum(pierwiastki.values()) == stopien:
            koniec = ''
            for ind, pierw in enumerate(pierwiastki.keys()):
                if ind == stopien:
                    pierw = str(pierw).strip('(').strip(')')
                    koniec += f"{pierw}"
                else:
                    pierw = str(pierw).strip('(').strip(')')
                    koniec += f"{pierw},"

            print('\n'+tekst + f" , pierwiastki: {koniec}")

            return 0

        try:
            new_f = lambda x: f(x) / (x - xn)
        except:
            new_f = lambda x: f(x) / (x - xn + epsilon)

        Df = lambda x: (new_f(x) - new_f(x-h)) / (h)

        x0 = a0 + complex(zero_vector / np.abs(zero_vector) * R)
        newton(new_f, Df, x0, epsilon, n_steps)

def residuum(R, a, n=1e3):
    '''Metoda służąca do określenia liczby pierwiastków wielomianu p(x) znajdujących się w obszarze 
    ograniczonym okręgiem o środku w (a,0) i promieniu R.'''

    def p(x):
        return np.sum([wspolczynniki[i]*np.power(x,i) for i in range(stopien+1)])

    def p_prime(x):
        return np.sum([i*wspolczynniki[i]*np.power(x,i-1) for i in range(1, stopien+1)])

    def P(x):
         return p_prime(x)/p(x)

    def gamma(x):
        '''Funkcja służąca do parametryzacji okręgu o środku w (a,0) i promieniu R'''
        return a + R*np.exp(1j*x) # a+ R*e^{it}

    def gamma_prime(x):
        '''Pochodna z funkcji parametryzującej okrąg'''
        return R*1j*np.exp(1j*x) # R*i*e^{it}
    fun = lambda x: P(gamma(x))*gamma_prime(x)/(2*np.pi*1j)
    # policz numerycznie calke po konturze

    h = (2*np.pi) / float(n)
    integral = 0

    for i in range(0, int(n)):
        integral = integral + 1/2 * h * (fun(i * h) + fun((i+1) * h))

    if round(np.real(integral)) < stopien:
        integral = 0
        R = R + float(np.random.random(1)/10)
        fun = lambda x: P(gamma(x))*gamma_prime(x)/(2*np.pi*1j)
        for i in range(0, int(n)):
            integral = integral + 1/2 * h * (fun(i * h) + fun((i+1) * h))

    return round(np.real(integral))
        

pierwiastki = {}
h = 1e-6

f = lambda x: np.sum([wspolczynniki[i] * np.power(x, i) for i in range(stopien+1)])
Df = lambda x: (f(x) - f(x-h)) / (h)

newton(f,Df,x0,epsilon=1e-8,n_steps=700)

with warnings.catch_warnings():
    warnings.filterwarnings('error')
    try:
        print("\nW obszarze ograniczonym okręgiem o środku w ({},0) i promieniu R={} znajdziesz {} z {} pierwiastków wielomianu.".format(a0, R, int(residuum(R, a=a0)), stopien))
    except Warning:
        print(f"\nCałka jest rozbieżna dla R={R}. Któryś z pierwiastków leży na konturze lub jest równy R. Spróbujmy nieco zwiększyć promień:")
        nieudana_proba = 1
        while(nieudana_proba):
            try:
                print("\nw obszarze ograniczonym okręgiem o środku w ({},0) i promieniu R={:.4f} znajdziesz {} z {} pierwiastków wielomianu.".format(a0, R+float(np.random.random(1))/10, int(residuum(R+1/10, a=a0)), stopien))
                nieudana_proba = 0
            except Warning:
                continue

# https://math.stackexchange.com/questions/2952073/finding-a-matrix-given-its-characteristic-polynomial

