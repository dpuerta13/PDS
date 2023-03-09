# Transformadas de Laplace, funciones
# http://blog.espol.edu.ec/telg1001/
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
equivalentes = [{'DiracDelta': lambda x: 1*(x==0)},
                {'Heaviside': lambda x,y: np.heaviside(x, 1)},
                'numpy',]

# variables y funciones simbólicas usadas
t = sym.Symbol('t',real=True)
tau = sym.Symbol('tau',real=True)
s = sym.Symbol('s')
z = sym.Symbol('z')
y = sym.Function('y')
x = sym.Function('x')
h = sym.Function('h')
u = sym.Heaviside(t)
d = sym.DiracDelta(t)

# UNIDAD 3

# Sistemas Lineales e invariantes en tiempo Sympy-Python
# http://blog.espol.edu.ec/telg1001/lti-ct-respuesta-entrada-cero-con-sympy-python/

def respuesta_ZIR(ecuacion,cond_inicio=[],t0=0,
                y = sym.Function('y'),x = sym.Function('x')):
    ''' Sympy: ecuacion: diferencial en forma(LHS,RHS),
        condiciones de inicio t0 y [y'(t0),y(t0)]
        cond_inicio en orden descendente derivada
    '''
    # ecuacion homogenea x(t)=0, entrada cero
    RHSx0 = ecuacion.rhs.subs(x(t),0).doit()
    LHSx0 = ecuacion.lhs.subs(x(t),0).doit()
    homogenea = LHSx0 - RHSx0

    # solucion general entrada cero
    general = sym.dsolve(homogenea,y(t))
    general = general.expand()

    # aplica condiciones iniciales 
    N = sym.ode_order(ecuacion,y(t)) # orden Q(D)
    eq_condicion = []
    for k in range(0,N,1):
        ck   = cond_inicio[(N-1)-k]
        dyk  = general.rhs.diff(t,k)
        dyk  = dyk.subs(t,t0)
        eq_k = sym.Eq(ck,dyk)
        eq_condicion.append(eq_k)
        
    constante = sym.solve(eq_condicion)

    # reemplaza las constantes en general
    y_c = general
    for Ci in constante:
        y_c = y_c.subs(Ci, constante[Ci])
    # respuesta a entrada cero ZIR
    ZIR = y_c.rhs
    
    sol_ZIR = {'homogenea'   : sym.Eq(homogenea,0),
               'general'     : general,
               'eq_condicion': eq_condicion,
               'constante'   : constante,
               'ZIR' : ZIR,}
    return(sol_ZIR)

def respuesta_impulso_h(ecuacion,t0=0,
                        y = sym.Function('y'),
                        x = sym.Function('x')):
    ''' respuesta a impulso h(t) de un
        sistema con Ecuacion Diferencial lineal
    '''
    # Método simplificado al emparejar términos
    N = sym.ode_order(ecuacion,y)
    M = sym.ode_order(ecuacion,x)

    # coeficiente de derivada de x(t) de mayor orden
    b0 = sym.nan
    if N>M:  # orden de derivada diferente
        b0 = 0
    if N==M: # busca coeficiente de orden mayor
        eq_RHS = sym.expand(ecuacion.rhs)
        term_suma = sym.Add.make_args(eq_RHS)
        for term_k in term_suma:
            # coeficiente derivada mayor
            if (M == sym.ode_order(term_k,x)): 
                b0 = 1 # para separar coeficiente
                factor_mul = sym.Mul.make_args(term_k)
                for factor_k in factor_mul:
                    if not(factor_k.has(sym.Derivative)):
                        b0 = b0*factor_k

    # Condiciones iniciales para respuesta a impulso
    cond_inicio    = [0]*N # lista de ceros tamano N
    cond_inicio[0] = 1     # condicion de mayor orden

    # ecuacion homogenea x(t)=0, entrada cero y
    # condiciones de impulso unitario
    sol_yc = respuesta_ZIR(ecuacion,cond_inicio)
    yc = sol_yc['ZIR']

    # Respuesta a impulso h(t)
    P_y = ecuacion.rhs.subs(x(t),yc).doit()
    h = P_y*u + b0*d

    sol_yc['N'] = N
    sol_yc['M'] = M
    sol_yc['cond_inicio'] = cond_inicio
    sol_yc['b0'] = b0
    sol_yc['h'] = h
    return(sol_yc)

def respuesta_ZSR(x,h):
    '''Respuesta a estado cero x(t) y h(t)
    '''
    # revisa causalidad de señales
    xcausal = es_causal(x)
    hcausal = es_causal(h)

    # intercambia si h(t) no es_causal
    # con x(t) es_causal por propiedad conmutativa
    intercambia = False
    if hcausal==False and xcausal==True:
        temporal = h
        h = x
        x = temporal
        xcausal = False
        hcausal = True
        intercambia = True

    # limites de integral de convolución
    tau_a = -sym.oo ; tau_b = sym.oo
    if hcausal==True:
        tau_b = t
    if (xcausal and hcausal)==True:
        tau_a = 0

    # integral de convolución x(t)*h(t)
    xh = x.subs(t,tau)*h.subs(t,t-tau)
    xh = sym.expand(xh,t)
    ZSR = sym.integrate(xh,(tau,tau_a,tau_b))
    ZSR = sym.expand(ZSR,t)
    if not(ZSR.has(sym.Integral)):
        ZSR = simplifica_escalon(ZSR)

    lista_escalon = ZSR.atoms(sym.Heaviside)
    ZSR = sym.expand(ZSR,t) # terminos suma
    ZSR = sym.collect(ZSR,lista_escalon)

    if intercambia == True:
        xcausal = True
        hcausal = False

    # graficar si no tiene Integral o error
    cond_graf = ZSR.has(sym.Integral)
    cond_graf = cond_graf or ZSR.has(sym.oo)
    cond_graf = cond_graf or ZSR.has(sym.nan)
    cond_graf = not(cond_graf)
    
    sol_ZSR = {'xh'      : xh,
               'xcausal' : xcausal,
               'hcausal' : hcausal,
               '[tau_a,tau_b]': [tau_a,tau_b],
               'intercambia'  : intercambia,
               'cond_graf'    : cond_graf,
               'ZSR' : ZSR,}
    return(sol_ZSR)

def busca_impulso(ft):
    ''' busca en f(t) impulsos sym.DiracDelta
        entrega una lista ordenada de resultados
    '''
    def impulso_donde(ft):
        ''' busca posicion de impulso en ft simple d, d**2
            un solo termino,sin factores
        '''
        ft = sym.sympify(ft,t) # convierte a sympy si es constante
        donde = [] # revisar f(t) sin impulsos
        if ft.has(sym.DiracDelta):
            if not(ft.is_Pow): # sin exponentes
                ecuacion = sym.Eq(ft.args[0],0)
            else: # con exponentes d**2
                ecuacion = sym.Eq(ft.args[0].args[0],0)
            donde = sym.solve(ecuacion,t)
        return(donde)
    
    def impulso_donde_unterm(ft):
        ''' revisa impulso en un termino suma de ft
        '''
        donde = [] # revisar f(t) sin impulsos
        factor_mul = sym.Mul.make_args(ft)
        for factor_k in factor_mul:
            if factor_k.has(sym.DiracDelta):
                donde_k = impulso_donde(factor_k)
                if len(donde) == 0: # vacio
                    donde.extend(donde_k)
                if len(donde)>0: # sin repetir
                    if not(donde_k[0] in donde): 
                        donde = [] # anulado por d(t-a)*d(t-b)
        return(donde)

    # revisa terminos suma
    ft = sym.sympify(ft,t) # convierte a sympy si es constante
    ft = sym.expand(ft)
    respuesta = []
    term_suma = sym.Add.make_args(ft)
    for term_k in term_suma:
        donde = impulso_donde_unterm(term_k)
        if not(donde in respuesta) and len(donde)>0:
            respuesta.extend(donde)
    if len(respuesta)>1: # ordena ascendente
        respuesta.sort()
    respuesta = list(respuesta)
    return(respuesta)

def busca_escalon(ft):
    ''' busca en f(t) el donde,sentido de escalon unitario
        en un termino simple  sin sumas, para ubicar
        lado del plano de f(t)
    '''
    
    def escalon_donde(ft):
        ''' ft sin factores o terminos suma
        '''
        ft = sym.sympify(ft,t) # convierte a sympy si es constante
        respuesta = []
        if ft.has(sym.Heaviside):
            eq_u      = sym.Eq(ft.args[0],0)
            sentido   = sym.diff(eq_u.lhs,t,1)
            donde     = sym.solve(eq_u,t)[0]
            respuesta = [donde,sentido]
        return(respuesta)

    def escalon_donde_unterm(ft):
        '''revisa termino de factores multiplica
        '''
        respuesta = []
        factor_mul = sym.Mul.make_args(ft)
        for factor_k in factor_mul:
            if factor_k.has(sym.Heaviside):
                ubicado = []
                if not(factor_k.is_Pow): # sin exponente
                    ubicado = escalon_donde(factor_k)
                else:  # con exponentes d**k
                    ubicado = escalon_donde(factor_k.args[0])
                if len(ubicado)>0:
                    respuesta.append(ubicado)
        return(respuesta)
        
    # revisa terminos suma
    ft = sym.sympify(ft,t) # convierte a sympy si es constante
    ft = sym.expand(ft)
    respuesta = []
    term_suma = sym.Add.make_args(ft)
    for term_k in term_suma:
        donde = escalon_donde_unterm(term_k)
        if not(donde in respuesta) and len(donde)>0:
            respuesta.extend(donde)
    if len(respuesta)>1: # ordena ascendente
        respuesta = np.array(respuesta)
        columna   = respuesta[:,0]
        ordena    = np.argsort(columna)
        respuesta = respuesta[ordena]

    return(respuesta)


def es_causal(ft):
    ''' h(t) es causal si tiene
        b0*d(t)+u(t)*(modos caracterisicos)
    '''
    def es_causal_impulso(ft):
        ''' un termino en lado derecho del plano
        '''
        causal  = False
        donde_d = busca_impulso(ft)
        if len(donde_d)>0:    # tiene impulso
            if donde_d[0]>=0: # derecha del plano
                causal = True 
        return(causal)
    
    causal  = True
    term_suma = sym.Add.make_args(ft)
    for term_k in term_suma:
        if term_k.has(sym.Heaviside):
            term_k = simplifica_escalon(term_k) # h(t-a)*h(t-b)
            causal_k = True
            donde_u  = busca_escalon(term_k)
            if len(donde_u)==0: # sin escalon?
                causal_k = False
            if len(donde_u)==1: 
                sentido1 = donde_u[0][0]
                donde1   = donde_u[0][1]
                # en lado izquierdo del plano
                if donde1<0 or sentido1<0:
                    causal_k = False
            if len(donde_u)==2:
                donde1   = donde_u[0][0]
                sentido1 = donde_u[0][1]
                donde2   = donde_u[1][0]
                sentido2 = donde_u[1][1]
                # rectangulo lado derecho del plano
                if (donde1<donde2): 
                    if donde1<0:  # u(t+1)*u(-t+1)
                        causal_k = False
                if (donde2<donde1):
                    if donde2<0: # u(-t+1)*u(t+1)
                        causal_k = False

        else: # un termino, sin escalon unitario
            # causal depende si tiene un impulso
            causal_k = es_causal_impulso(term_k)
        causal = causal and causal_k
    return(causal)

def simplifica_impulso(ft):
    ''' simplifica d**2, d(t-1)*d
    '''
    def simplifica_d_d(ft):
        '''un termino de suma  d**2, d(t+1)*d,
        '''
        respuesta = ft
        if ft.has(sym.DiracDelta):# tiene impulsos
            impulso_en = busca_impulso(ft)
            if len(impulso_en)==0: # anulado por d(t-a)*d(t-b)
               respuesta = 0*t
            elif len(impulso_en)>0: # tiene impulsos
                respuesta = 1
                factor_mul = sym.Mul.make_args(ft)
                for factor_k in factor_mul:
                    if not(factor_k.has(sym.DiracDelta)):
                        if not(factor_k.has(sym.Heaviside)):
                            termino = factor_k.subs(t,impulso_en[0])
                        else: # tiene escalón
                            termino = factor_k.subs(t,impulso_en[0])
                            if termino ==1/2:
                                termino = 1
                        respuesta = respuesta*termino
                    else:  # factor con impulso
                        if factor_k.is_Pow: # tiene exponente
                            respuesta = respuesta*factor_k.args[0]
                        else: # termino sin exponente
                            respuesta = respuesta*factor_k
        return(respuesta)
    
    # revisa terminos suma
    respuesta = 0*t
    ft = sym.expand(ft,t)
    term_suma = sym.Add.make_args(ft)
    for term_k in term_suma:
        respuesta = respuesta + simplifica_d_d(term_k)
        
    return(respuesta)

def simplifica_escalon(ft):
    ''' simplifica multiplicaciones
        Heaviside(t-a)*Heaviside(t-b) en f(t)
    '''
    def simplifica_u_u(ft):
        '''solo dos pares de u(t-1)*u(t-2),
           sin coeficientes
        '''
        donde_u = busca_escalon(ft)
        donde_u = np.array(donde_u)
        # direccion donde_[:,1],
        # lugar donde[:,0]
        # analiza multiplicación
        resultado = ft   # todo igual
        if donde_u[0,1]*donde_u[1,1] > 0: # direccion igual
            k = 0
            if donde_u[0,1]>0: # hacia derecha
                k = np.argmax(donde_u[:,0])
                k_signo = 1
            else: # hacia izquierda
                k = np.argmin(donde_u[:,0])
                k_signo = -1
            ubica = donde_u[k,1]*t-k_signo*donde_u[k][0]
            resultado = sym.Heaviside(ubica)
        else: # direccion diferente
            if donde_u[0][1]>0 and (donde_u[0,0]>donde_u[1,0]):
                    resultado = 0
            if donde_u[0][1]<0 and (donde_u[0,0]<=donde_u[1,0]):
                    resultado = 0
        return(resultado)

    def simplifica_u_term(ft):
        ''' simplifica un termino de varios
            factores que multiplican 2*pi*u*u(t-1)
        '''
        respuesta = ft
        if ft.has(sym.Heaviside): # tiene escalon
            escalon_en = busca_escalon(ft)
            revisa = 1 ; otros = 1 ; cuenta = 0
            factor_mul = sym.Mul.make_args(ft)
            for factor_k in factor_mul:
                if factor_k.has(sym.Heaviside):
                    if factor_k.is_Pow: # con exponente
                        revisa = revisa*factor_k.args[0]
                        cuenta = cuenta + 1
                    else: # sin exponente
                        revisa = revisa*factor_k
                        cuenta = cuenta + 1
                    if cuenta>1: # simplificar
                        revisa = simplifica_u_u(revisa)
                        cuenta = len(busca_escalon(revisa))
                else: # factor sin Heaviside
                    otros = otros*factor_k
            respuesta = otros*revisa

        return(respuesta)

    # revisa terminos suma
    respuesta = 0*t
    ft = sym.expand(ft,t)
    if ft.has(sym.DiracDelta): # tiene impulsos
        ft = simplifica_impulso(ft)
    term_suma = sym.Add.make_args(ft)
    for term_k in term_suma:
        respuesta = respuesta + simplifica_u_term(term_k)

    return(respuesta)

# UNIDAD 4
# Transformada de Laplace para f(t) con Sympy-Python
# http://blog.espol.edu.ec/telg1001/transformada-de-laplace-para-ft-con-sympy-python/
def laplace_integral_sumas(ft,unilateral=True):
    ''' integral de transformada de laplace
        considera impulsos d(t), escalon u(t) desplazados
    '''
    fts = ft*sym.exp(-s*t) # f(t,s) para integrar
    fts = sym.expand(fts)  # expresion de sumas
    fts = sym.powsimp(fts) # simplifica exponentes

    # Integral del impulso en cero es 1
    if fts.has(sym.DiracDelta): 
        donde = busca_impulso(fts)
        if (0 in donde) and unilateral: # (integral unilateral) x 2
            fts = fts.subs(d,2*d)
        fts = sym.powsimp(fts) # simplifica exponentes

    # Integral Laplace de sumas
    lim_a = 0 # unilateral
    if not(unilateral):
        lim_a = -sym.oo
    Fs = 0*s
    term_suma = sym.Add.make_args(fts)
    for term_k in term_suma:
        # integral Laplace unilateral
        Fs_L = sym.integrate(term_k,(t,lim_a,sym.oo))
        if Fs_L.is_Piecewise:   # Fs_L es por partes
            Fsk = Fs_L.args[0]  # primera ecuacion e intervalo
            Fsk = Fsk[0]        # solo expresion
        else: # una sola expresión
            Fsk = Fs_L   
        Fs = Fs + Fsk

    # convierte a Sympy si es solo constante
    Fs = sym.sympify(Fs)
    # lista los sym.exp(-s)
    lista_exp = list(Fs.atoms(sym.exp(-s)))

    # agrupa por cada exp(-s) en lista_exp
    if len(lista_exp)>0:
        Fs = sym.expand(Fs,s)
        Fs = sym.collect(Fs,lista_exp)
    else:
        Fs = sym.factor(Fs,s)
    return(Fs)

def laplace_transform_suma(ft):
    '''transformada de Laplace de suma de terminos
       separa constantes para conservar en resultado
    '''
    def separa_constante(termino):
        ''' separa constante antes de usar
            sym.laplace_transform(term_suma,t,s)
            para incorporarla luego de la transformada.
            inconveniente revisado en version 1.11.1
        '''
        constante = 1
        if termino.is_Mul:
            factor_mul = sym.Mul.make_args(termino)
            for factor_k in factor_mul:
                if not(factor_k.has(t)):
                    constante = constante*factor_k
            termino = termino/constante
        return([termino,constante])

    # transformadas de Laplace por términos suma
    ft = sym.expand(ft)  # expresion de sumas
    ft = sym.powsimp(ft) # simplifica exponentes

    term_suma = sym.Add.make_args(ft)
    Fs = 0
    for term_k in term_suma:
        [term_k,constante] = separa_constante(term_k)
        Fsk = sym.laplace_transform(term_k,t,s)
        Fs  = Fs + Fsk[0]*constante
        
    # separa exponenciales constantes
    Fs = sym.expand_power_exp(Fs)
    return(Fs)
def apart_exp(Fs):
    '''expande Fs en fracciones parciales
       considera términos con sym.exp multiplicados
    '''
    # convierte a Sympy si es solo constante
    Fs = sym.sympify(Fs)
    
    # simplifica operación H(s)*x(s)
    Fs = sym.expand_power_exp(Fs)
    Fs = sym.expand(Fs, powe_exp=False)
    # reagrupa Fs
    term_sum = sym.Add.make_args(Fs)
    Fs_0 = 0*s
    for term_k in term_sum:
        term_k = sym.simplify(term_k)
        term_k = sym.expand_power_exp(term_k)
        Fs_0 = Fs_0 + term_k
    
    # tabula sym.exp(-s) en lista_exp
    lista_exp = list(Fs_0.atoms(sym.exp(-s)))
    if len(lista_exp)>0: # elimina constantes
        for k in lista_exp:
            if not(k.has(s)): # es constante
                lista_exp.remove(k)

    # separados por lista_exp
    separados = sym.collect(Fs_0,lista_exp,evaluate=False)
    
    # fracciones parciales
    Fs_fp = 0
    for k in separados:
        Fs_k = sym.apart(separados[k],s)
        Fs_fp = Fs_fp + k*Fs_k
    return(Fs_fp)

def factor_exp(Fs):
    '''expande Fs en factores por sym.exp(-a*s)
       considera términos con sym.exp multiplicados
    '''
    # convierte a Sympy si es solo constante
    Fs = sym.sympify(Fs)
    
    # simplifica operación H(s)*x(s)
    Fs = sym.expand_power_exp(Fs)
    Fs = sym.expand(Fs, powe_exp=False)
    # reagrupa Fs
    term_sum = sym.Add.make_args(Fs)
    Fs_0 = 0*s
    for term_k in term_sum:
        term_k = sym.simplify(term_k)
        term_k = sym.expand_power_exp(term_k)
        Fs_0 = Fs_0 + term_k
    
    # tabula sym.exp(-s) en lista_exp
    lista_exp = list(Fs_0.atoms(sym.exp(-s)))
    if len(lista_exp)>0: # elimina constantes
        for k in lista_exp:
            if not(k.has(s)): # es constante
                lista_exp.remove(k)

    # separados por lista_exp
    separados = sym.collect(Fs_0,lista_exp,evaluate=False)
    
    # factores por lista_exp
    Fs_fc = 0
    for k in separados:
        Fs_k = sym.factor(separados[k],s)
        Fs_fc = Fs_fc + k*Fs_k
    
    return(Fs_fc)


def Q_cuad_parametros(Hs):
    '''Si Q tiene grado=2, obtiene los parametros
       para la tabla de Transformadas Inversas de Laplace
    '''
    def Q_cuad_sin_exp(Hs):
        '''para Hs sin sym.exp(-a*s)
        '''
        # fracciones parciales NO separa Q con factores complejos
        Hs_fp = sym.apart(Hs,s)

        respuesta = {}
        term_suma = sym.Add.make_args(Hs_fp)
        for term_k in term_suma:
            [P,Q]  = term_k.as_numer_denom()
            if sym.degree(Q) == 2 and sym.degree(P) == 1:
                P_coef = P.as_coefficients_dict(s)
                Q_coef = Q.as_coefficients_dict(s)
                Q0 = 1 # Normalizar coefficiente s**2=1
                if Q_coef[s**2]!=1: 
                    Q0 = Q_coef[s**2]
                # Parametros de Q cuadratico
                a = 0 ; c = 0
                if s**1 in Q_coef:
                    a = float(Q_coef[s**1]/Q0)/2
                if s**0 in Q_coef:
                    c = float(Q_coef[s**0]/Q0)
                A = float(P_coef[s**1])
                B = 0
                if s**0 in P_coef:
                    B = float(P_coef[s**0])
                rP = (A**2)*c + B**2 - 2*A*B*a
                rQ = c - a**2
                r  = np.sqrt(rP/rQ)
                b  = np.sqrt(c-a**2)
                thetaP = A*a-B
                thetaQ = A*np.sqrt(c-a**2)
                theta  = np.arctan(thetaP/thetaQ)
                parametro = {'A': A, 'B': B,'a': a, 'c': c,
                             'r': r, 'b': b,'theta':theta}
                respuesta[term_k] = parametro
        return (respuesta)

    Hs = sym.sympify(Hs,s)
    Hs_fp = apart_exp(Hs) # separa por sym.exp(-a*s)
    # tabula sym.exp(-s) en lista_exp
    lista_exp = list(Hs_fp.atoms(sym.exp(-s)))
    if len(lista_exp)>0: # elimina constantes
        for k in lista_exp:
            if not(k.has(s)): # es constante
                lista_exp.remove(k)

    # separados por lista_exp
    separados = sym.collect(Hs_fp,lista_exp,evaluate=False)

    Qs2 = {} # agrupa parametros Qs2
    for k in separados:
        Qs2_k = Q_cuad_sin_exp(separados[k])
        if len(Qs2_k)>0:
            for term in Qs2_k:
                Qs2[k*term] = Qs2_k[term]
    return(Qs2)

def busca_polosceros(Hs):
    ''' Busca polos de Hs (divisiones para cero)
        y ceros de Hs,cuenta las veces que aparecen,
        agrupa por exp(-a*s) agrupado en factores.
    '''
    # revisa dominio z para cambio de variable
    dominio_var = 's'
    z = sym.Symbol('z')
    if z in Hs.free_symbols:
        Hs = Hs.subs(z,s)
        dominio_var = 'z'
    def polosceros_simple(Hs):
        ''' Busca_polo de un termino sin exp(-a*s)
        '''
        Hs = sym.simplify(Hs,inverse=True)
        # Hs debería ser un solo termino suma
        Hs_factor = sym.factor(Hs,s)
        # polos y ceros de termino Hs
        [P,Q] = Hs_factor.as_numer_denom()
        P = sym.poly(P,s)  # numerador
        Q = sym.poly(Q,s)  # denominador
        P_ceros = sym.roots(P)
        Q_polos = sym.roots(Q)
        respuesta = {'Q_polos' : Q_polos,
                     'P_ceros' : P_ceros}
        return(respuesta)
    # convierte a Sympy si es solo constante
    Hs = sym.sympify(Hs)
    # simplifica operación H(s)*x(s)
    Hs = sym.expand_power_exp(Hs)
    Hs = sym.expand(Hs, power_exp=False)
    
    term_sum = sym.Add.make_args(Hs)
    Hs_0 = 0*s # reagrupa Fs
    for term_k in term_sum:
        term_k = sym.simplify(term_k)
        term_k = sym.expand_power_exp(term_k)
        Hs_0 = Hs_0 + term_k
    
    # tabula sym.exp(-s) en lista_exp
    lista_exp = list(Hs_0.atoms(sym.exp(-s)))
    if len(lista_exp)>0: # elimina constantes
        for k in lista_exp:
            if not(k.has(s)): # es constante
                lista_exp.remove(k)

    # separados por lista_exp
    separados = sym.collect(Hs_0,lista_exp,evaluate=False)

    # polos y ceros por terminos exp(-s) agrupados
    polosceros = {} ; Hs_fp = 0
    for k in separados:
        Hs_k = sym.factor(separados[k],s)
        PZ_k = polosceros_simple(Hs_k)
        PZ_k['Hs_k']  = Hs_k
        polosceros[k] = PZ_k

    # integra polos
    Q_polos = {} ; P_ceros = {}
    for k in polosceros:
        polos = polosceros[k]['Q_polos']
        for unpolo in polos:
            if unpolo in Q_polos:
                veces = max([Q_polos[unpolo],polos[unpolo]])
                Q_polos[unpolo] = veces
            else:
                Q_polos[unpolo] = polos[unpolo]
        ceros = polosceros[k]['P_ceros']
        for uncero in ceros:
            if uncero in P_ceros:
                veces = max([P_ceros[uncero],ceros[uncero]])
                P_ceros[unpolo] = veces
            else:
                P_ceros[uncero] = ceros[uncero]
    # revisa exp(-a*s)
    if len(lista_exp)==0: #sin componentes
        del polosceros[1]
    polosceros['Q_polos'] = Q_polos
    polosceros['P_ceros'] = P_ceros
    return(polosceros)

# LTI CT Laplace – H(s) polos y Estabilidad del sistema con Sympy-Python
# http://blog.espol.edu.ec/telg1001/lti-ct-laplace-hs-estabilidad-del-sistema/

def estabilidad_asintotica(Q_polos, casi_cero=1e-8):
    ''' Analiza estabilidad asintotica con Q_raiz
        Separa parte real e imaginaria de raices
        casicero es la tolerancia para considerar cero
    '''
    cuenta_real = 0; cuenta_imag = 0
    unicos = 0 ; repetidos = 0 ; enRHP = 0
    for raiz in Q_polos:
        [r_real,r_imag] = raiz.as_real_imag()
        if abs(r_real)>casi_cero and abs(r_imag)<casi_cero :
            cuenta_real = cuenta_real+1
        # para estabilidad asintotica
        conteo = Q_polos[raiz]
        if conteo==1 and r_real==0 and abs(r_imag)>0:
            unicos = unicos + 1
        if conteo>1  and r_real==0 and abs(r_imag)>0:
            repetidos = repetidos + 1
        if r_real>0:
            enRHP = enRHP + 1
    cuenta_imag = len(Q_polos)-cuenta_real

    # Revisa lado derecho del plano RHP
    asintota = ""
    if enRHP==0:
        asintota = 'estable'
    if enRHP>0 or repetidos>0:
        asintota = 'inestable'
    if enRHP==0 and unicos>0:
        asintota = 'marginalmente estable'

    estable = {'n_polos_real': cuenta_real,
               'n_polos_imag': cuenta_imag,
               'enRHP'     : enRHP,
               'unicos'    : unicos,
               'repetidos' : repetidos,
               'asintota'  : asintota,}
    return(estable)

# LTI CT Laplace – Y(s)=ZIR+ZSR Respuesta entrada cero y
# estado cero con Sympy-Python
# http://blog.espol.edu.ec/telg1001/lti-ct-laplace-yszirzsr-respuesta-entrada-cero-y-estado-cero/
def respuesta_ZIR_s(Hs,cond_inicio):
    '''respuesta a entrada cero ZIR con H(s) y condiciones de inicio
       Si ZIR es sym.nan existen insuficientes condiciones
    '''
    polosceros = busca_polosceros(Hs)
    Q_polos = polosceros['Q_polos']
    
    # coeficientes y grado de Q
    Q = 1+0*s
    for raiz in Q_polos:
        veces = Q_polos[raiz]
        Q = Q*((s-raiz)**veces)
    Q = sym.poly(Q,s)
    
    Q_coef  = Q.all_coeffs() # coeficientes Q
    Q_grado = Q.degree()  # grado polinomio Q

    ZIR_Qs2 = {}
    term_cero = 0*s 
    if len(cond_inicio) == Q_grado:
        for j in range(0,Q_grado,1):
            term_orden = 0
            for i in range(j,Q_grado,1):
                term_orden = term_orden + cond_inicio[i]*(s**(i-j))
            term_cero = term_cero + term_orden*Q_coef[j]
        ZIR = term_cero/Q
        ZIR = apart_exp(ZIR)
        if not(ZIR==0): # no es constante o cero
            ZIR_Qs2 = Q_cuad_parametros(ZIR)
    else:
        ZIR = sym.nan # insuficientes condiciones iniciales

    if not(ZIR==sym.nan):
        # entrada_cero en t
        yt_ZIR = sym.inverse_laplace_transform(ZIR,s,t)
        # simplifica log(exp()) ej: e**(-2t)/(s**2)
        if yt_ZIR.has(sym.log):
            yt_ZIR = sym.simplify(yt_ZIR,inverse=True)
        lista_escalon = yt_ZIR.atoms(sym.Heaviside)
        yt_ZIR = sym.expand(yt_ZIR,t) # terminos suma
        yt_ZIR = sym.collect(yt_ZIR,lista_escalon)
        
        sol_ZIR ={'term_cero' : term_cero}
        if len(ZIR_Qs2)>0: # añade si Q es cuadratico
            sol_ZIR['ZIR_Qs2'] = ZIR_Qs2
        sol_ZIR = sol_ZIR | {'ZIR'   : ZIR,
                             'yt_ZIR': yt_ZIR}
    else:
        sol_ZIR = sym.nan
    
    return(sol_ZIR)

def respuesta_ZSR_s(Hs,Xs):
    '''respuesta a estado cero ZSR con H(s) y X(s)
    '''
    ZSR = Hs*Xs
    ZSR = apart_exp(ZSR)
    ZSR_Qs2 = Q_cuad_parametros(ZSR)

    # ZSR respuesta estado cero(t)
    yt_ZSR = sym.inverse_laplace_transform(ZSR,s,t)

    # simplifica log(exp()) ej: e**(-2t)/(s**2)
    if yt_ZSR.has(sym.log):
        yt_ZSR = sym.simplify(yt_ZSR,inverse=True)
    lista_escalon = yt_ZSR.atoms(sym.Heaviside)
    yt_ZSR = sym.expand(yt_ZSR,t) # terminos suma
    yt_ZSR = sym.collect(yt_ZSR,lista_escalon)

    sol_ZSR = {'ZSR' : ZSR}
    if len(ZSR_Qs2)>0:
        sol_ZSR['ZSR_Qs2'] = ZSR_Qs2
    sol_ZSR = sol_ZSR | {'yt_ZSR' : yt_ZSR}
    return(sol_ZSR)

# Graficas de ejercicios
def intervalo_s(Q_polos={},P_ceros={}):
    '''estima intervalo para s usando Q_polos y P_ceros
    '''
    s_a = 0 ; s_b = 0
    if len(Q_polos)>0:  # si existen raíces
        for raiz in Q_polos:
            una_raiz = sym.sympify(raiz)
            [r_real,r_imag] = una_raiz.as_real_imag()
            s_a = min(s_a,float(r_real))
            s_b = max(s_b,float(r_real))
    if len(P_ceros)>0:  # si existen raíces
        for raiz in P_ceros:
            una_raiz = sym.sympify(raiz)
            [r_real,r_imag] = una_raiz.as_real_imag()
            s_a = min(s_a,float(r_real))
            s_b = max(s_b,float(r_real))
    s_a = int(s_a) - 1
    s_b = int(s_b) + 1
    if s_a>-1:
        s_a = -1
    if s_b<=0: 
        s_b = 1
    return([s_a,s_b])

def graficar_xhy(xt,ht,yt,t_a,t_b,
                   muestras=101,x_nombre='x',
                   h_nombre='h',y_nombre='y'):
    # grafica valores para x(t),h(t),y(t)
    ti  = np.linspace(t_a,t_b,muestras)

    xt = sym.sympify(xt,t) # convierte a sympy una constante
    if xt.has(t): # no es constante
        x_t = sym.lambdify(t,xt,modules=equivalentes)
    else:
        x_t = lambda t: xt + 0*t
    xti = x_t(ti)
    impulsoen_xt = busca_impulso(xt)

    ht = sym.sympify(ht,t) # convierte a sympy una constante
    if ht.has(t): # no es constante
        h_t = sym.lambdify(t,ht,modules=equivalentes)
    else:
        h_t = lambda t: ht + 0*t
    hti = h_t(ti)
    impulsoen_ht = busca_impulso(ht)

    yt = sym.sympify(yt,t) # convierte a sympy una constante
    if yt.has(t): # no es constante
        y_t = sym.lambdify(t,yt,modules=equivalentes)
    else:
        y_t = lambda t: yt + 0*t
    yti = y_t(ti)
    impulsoen_yt = busca_impulso(yt)
    
    # grafica x(t),h(t),y(t)
    figura, graf_fti = plt.subplots()
    colorlinea_x = 'blue'
    if x_nombre =='ZIR':
        colorlinea_x = 'orange'
    if x_nombre =='ZSR':
        colorlinea_x = 'dodgerblue'
    graf_fti.axvline(0, color='gray')
    graf_fti.axhline(0, color='gray')
    graf_fti.plot(ti,xti,color = colorlinea_x,
                  label=x_nombre+'(t)')
    # Graficar terminos con impulso
    if len(impulsoen_xt)>0:
        for donde in impulsoen_xt:
            tk = float(donde)
            yk = float(f_xt(tk))
            graf_fti.stem(tk,yk)

    # Grafica h(t)
    linestyle_h='dashed'
    if h_nombre!='h':
        linestyle_h='solid'
    colorlinea_h = 'magenta'
    if h_nombre =='ZIR':
        colorlinea_h = 'orange'
    if h_nombre =='ZSR':
        colorlinea_h = 'dodgerblue'
    graf_fti.plot(ti,hti,label=h_nombre+'(t)',
             linestyle=linestyle_h,
             color=colorlinea_h)
    if len(impulsoen_ht)>0:
        for donde in impulsoen_ht:
            tk = float(donde)
            yk = float(f_ht(tk))
            graf_fti.stem(tk,yk,
                          linefmt=colorlinea_h)
    # Grafica y(t)
    colorlinea_y = 'green'
    if y_nombre =='ZSR':
        colorlinea_y = 'dodgerblue'
    if y_nombre =='ZIR':
        colorlinea_y = 'orange'
        
    graf_fti.plot(ti,yti,label=y_nombre+'(t)',
                  color=colorlinea_y)
    # Graficar terminos con impulso
    if len(impulsoen_yt)>0:
        for donde in impulsoen_yt:
            tk = float(donde)
            yk = float(f_xt(tk))
            graf_fti.stem(tk,yk,
                          linefmt=colorlinea_y)
    graf_fti.set_xlabel('t')
    graf_fti.set_ylabel('f(t)')
    etiqueta = r''+y_nombre+'(t) = $'
    etiqueta = etiqueta +str(sym.latex(yt))+'$'
    graf_fti.set_title(etiqueta)
    graf_fti.legend()
    graf_fti.grid()
    return(graf_fti)

def graficar_ft(ft,t_a,t_b,muestras=101,f_nombre='f'):
    ''' Graficar f(t) en intervalo [t_a,t_b]
        f_nombre: x,y,ZIR,h,ZSR
    '''
    # calcula f(t) en intervalo
    ft = sym.sympify(ft,t) # convierte a sympy una constante
    if ft.has(t): # no es constante
        f_t = sym.lambdify(t,ft,modules=equivalentes)
    else:
        f_t = lambda t: ft + 0*t
    ti  = np.linspace(t_a,t_b,muestras)
    fti = f_t(ti)
    # grafica parametros
    figura_t, graf_ft = plt.subplots()
    graf_ft.axvline(0, color='gray')
    graf_ft.axhline(0, color='gray')
    colorlinea = 'blue'
    if f_nombre=='x':
        colorlinea = 'blue'
    if f_nombre=='y':
        colorlinea = 'green'
    if f_nombre=='ZIR':
        colorlinea = 'orange'
    if f_nombre=='h':
        colorlinea = 'magenta'
    if f_nombre=='ZSR':
        colorlinea = 'dodgerblue'

    # grafica de f(t)
    graf_ft.plot(ti,fti,label=f_nombre+'(t)',
                 color=colorlinea)

    # Graficar terminos con impulso
    impulsoen_ft = busca_impulso(ft)
    if len(impulsoen_ft)>0:
        for donde in impulsoen_ft:
            tk = float(donde)
            yk = float(f_t(tk))
            graf_ft.stem(tk,yk,
                         linefmt=colorlinea)
    
    graf_ft.set_xlabel('t')
    graf_ft.set_ylabel(f_nombre+'(t)')
    etiqueta = r''+f_nombre+'(t) = $'
    etiqueta = etiqueta+str(sym.latex(ft))+'$'
    graf_ft.set_title(etiqueta)
    graf_ft.legend()
    graf_ft.grid()
    return(figura_t)

def graficar_Fs2(Fs,Q_polos={},P_ceros={},
                s_a=1,s_b=0,muestras=101,f_nombre='F',
                solopolos=False,polosceros={}):
    # grafica Fs
    lista_Hs =[]

    lista_PZ = ['Q_polos','P_ceros']
    lista_term = list(polosceros.keys())
    lista_term.remove('Q_polos')
    lista_term.remove('P_ceros')

    # intervalo para s,y componentes Fs con exp(-a*s)
    s_a = 0 ; s_b = 0 ; Lista_Hs = []
    cond_componente = False
    if len(lista_term)>0:
        cond_componente = True
        if len(polosceros)>1: # tiene componenes con exp(-a*s)
            lista_Hs.append([Hs,{},{}])
        for k in polosceros:
            if not(k in lista_PZ):
                Q_polosk = polosceros[k]['Q_polos']
                P_cerosk = polosceros[k]['P_ceros']
                [s_ak,s_bk] = intervalo_s(Q_polosk,P_cerosk)
                s_a = min(s_a,s_ak)
                s_b = max(s_b,s_bk)
                lista_Hs.append([polosceros[k]['Hs_k']*sym.sympify(k),
                                 Q_polosk,P_cerosk])
    else: #sin componenes con exp(-a*s)
        lista_Hs.append([Hs,polosceros['Q_polos'],polosceros['P_ceros']])
        [s_a,s_b] = intervalo_s(polosceros['Q_polos'],
                                       polosceros['P_ceros'])
    # graficas por cada componente
    fig_Fs, graf_Fs = plt.subplots()
    # no presenta errores de división para cero en lambdify()
    np.seterr(divide='ignore', invalid='ignore')
    for k in lista_Hs:
        Fs_k = k[0] ;Q_polosk = k[1] ;P_cerosk = k[2]
        
        # convierte a sympy una constante
        Fs_k = sym.sympify(Fs_k,s) 
        if Fs_k.has(s): # no es constante
            F_s = sym.lambdify(s,Fs_k,modules=fcnm.equivalentes)
        else:
            F_s = lambda s: Fs_k + 0*s
        
        s_i = np.linspace(s_a,s_b,muestras)
        Fsi = F_s(s_i) # Revisar cuando s es complejo
        lineaestilo = 'solid'
        if len(Q_polosk)>0 and cond_componente:
            lineaestilo = 'dashed'
            
        graf_Fs.plot(s_i,Fsi,label=Fs_k,linestyle=lineaestilo)
        lineacolor = plt.gca().lines[-1].get_color()

        if len(Q_polosk)>0:
            polo_re = [] ; polo_im = []
            for polos in Q_polosk.keys():    
                graf_Fs.axvline(sym.re(polos),color='red',
                                linestyle='dotted')
                x_polo = sym.re(polos)
                y_polo = sym.im(polos)
                polo_re.append(x_polo)
                polo_im.append(y_polo)
                etiqueta = "("+str(x_polo)+','+str(y_polo)+")"
                graf_Fs.annotate(etiqueta,(x_polo,y_polo))
        
            graf_Fs.scatter(polo_re,polo_im,marker='x',
                            color =lineacolor,
                            label = Q_polosk)
        if len(P_cerosk)>0:
            cero_re = [] ; cero_im = []
            for cero in P_cerosk.keys():    
                x_cero = sym.re(cero)
                y_cero = sym.im(cero)
                cero_re.append(x_cero)
                cero_im.append(y_cero)
                etiqueta = "("+str(x_cero)+','+str(y_cero)+")"
                graf_Fs.annotate(etiqueta,(x_cero,y_cero))
        
            graf_Fs.scatter(cero_re,cero_im,marker='o',
                            color =lineacolor,
                            label = P_cerosk)

    graf_Fs.axvline(0,color='gray')
    graf_Fs.legend()
    graf_Fs.set_xlabel('Real(s)')
    graf_Fs.set_ylabel('F(s) ; Imag(s)')
    graf_Fs.set_title(r'F(s) = $'+str(sym.latex(Hs))+'$')
    graf_Fs.grid()
    return(fig_Fs)

def graficar_Fs(Fs,Q_polos={},P_ceros={},
                s_a=1,s_b=0,muestras=0,f_nombre='F',
                solopolos=False):
    ''' grafica de Fs con polos y ceros.
        calcula intervalo si no se da [s_a,s_b]
        Las muestras si no se proporcionan se calculan
        como 10 muestras por unidad para el intervalo o 101.
    '''
    # revisa dominio z para cambio de variable
    dominio_var = 's'
    z = sym.Symbol('z')
    if z in Fs.free_symbols:
        Fz = Fs
        Fs = Fs.subs(z,s)
        dominio_var = 'z'
    # dominio s, corte en plano real
    if (s_a>s_b) and len(Q_polos)>0:
        [s_a,s_b] = intervalo_s(Q_polos,P_ceros)
        muestras_k = int(s_b-s_a)*10+1
        if muestras == 0:
            muestras = muestras_k
    else:
        s_a = -1 ; s_b =1
    if muestras == 0:
        muestras = 101
    # grafica evaluación numerica
    np.seterr(divide='ignore', invalid='ignore')
    Fs = sym.sympify(Fs,s) # convierte a sympy una constante
    if Fs.has(s): # no es constante
        F_s = sym.lambdify(s,Fs,modules=equivalentes)
    else:
        F_s = lambda s: Fs + 0*s
    
    s_i = np.linspace(s_a,s_b,muestras)
    Fsi = F_s(s_i)

    color_y = 'green'
    color_polo = 'red'
    color_cero = 'blue'

    figura_s, graf_Fs = plt.subplots()
    if solopolos==False:
        etiqueta = f_nombre+'('+str(dominio_var)+')'
        graf_Fs.plot(s_i,Fsi,label=etiqueta
                     ,color = color_y)
    for raiz in Q_polos.keys():
        x_polo = sym.re(raiz)
        y_polo = sym.im(raiz)
        graf_Fs.scatter(x_polo,y_polo,
                    marker='x',color=color_polo,
                    label='polo:'+str(raiz))
        graf_Fs.axvline(x_polo,color=color_polo,
                    linestyle='dashed')
        
        etiqueta = "("+str(np.around(float(x_polo),4))
        etiqueta = etiqueta + ','+str(np.around(float(y_polo),4))+")"
        graf_Fs.annotate(etiqueta,(x_polo,y_polo),rotation=45)
    for raiz in P_ceros.keys():
        x_cero = sym.re(raiz)
        y_cero = sym.im(raiz)
        graf_Fs.scatter(x_cero,y_cero,
                    marker='o',color=color_cero,
                    label='cero:'+str(raiz))
        graf_Fs.axvline(x_cero,color=color_cero,
                    linestyle='dotted')
        etiqueta = "("+str(np.round(float(x_cero),4))+','+str(np.around(float(y_cero),4))+")"
        graf_Fs.annotate(etiqueta,(x_cero,y_cero),rotation=45)
    graf_Fs.axvline(0, color='gray')
    graf_Fs.axhline(0, color='gray')
    graf_Fs.set_xlabel('Re('+str(dominio_var)+')')
    etiqueta = 'Imag('+str(dominio_var)+')'
    if solopolos==0:
        etiqueta = f_nombre+'('+str(dominio_var)+') ; '+etiqueta
    graf_Fs.set_ylabel(etiqueta)
    etiqueta = r''+f_nombre+'('+str(dominio_var)+') = $'
    if dominio_var=='s':
        etiqueta = etiqueta+str(sym.latex(Fs))+'$'
    elif dominio_var=='z':
        etiqueta = etiqueta+str(sym.latex(Fz))+'$'
    graf_Fs.set_title(etiqueta)
    graf_Fs.legend()
    graf_Fs.grid()
    return(graf_Fs)

def graficar_xh_y(xt,ht,yt,t_a,t_b,
                  muestras=101,x_nombre='x',
                  h_nombre='h',y_nombre='y'):
    '''dos subgraficas, x(t) y h(t) en superior
       h(t) en inferior
    '''
    
    # grafica evaluación numerica
    xt = sym.sympify(xt,t) # convierte a sympy una constante
    if xt.has(t): # no es constante
        x_t = sym.lambdify(t,xt,modules=equivalentes)
    else: # es constante 
        x_t = lambda t: xt + 0*t

    ht = sym.sympify(ht,t) # convierte a sympy una constante
    if ht.has(t): # no es constante
        h_t = sym.lambdify(t,ht,modules=equivalentes)
    else: # es constante 
        h_t = lambda t: ht + 0*t

    ti = np.linspace(t_a,t_b,muestras)
    xi = x_t(ti)
    hi = h_t(ti)

    yt = sym.sympify(yt,t) # convierte a sympy una constante
    if yt.has(t): # no es constante
        y_t = sym.lambdify(t,yt,modules=equivalentes)
    else: # es constante 
        y_t = lambda t: yt + 0*t
    yi = y_t(ti)

    colorlinea_y = 'green'
    if y_nombre =='ZSR':
        colorlinea_y = 'dodgerblue'
    
    fig_xh_y, graf2 = plt.subplots(2,1)
    untitulo = y_nombre+'(t) = $'+ str(sym.latex(yt))+'$'
    graf2[0].set_title(untitulo)
    graf2[0].plot(ti,xi, color='blue', label='x(t)')
    graf2[0].plot(ti,hi, color='magenta', label='h(t)')
    graf2[0].legend()
    graf2[0].grid()
    # Graficar terminos con impulso
    impulsoen_x = busca_impulso(xt)
    if len(impulsoen_x)>0:
        for donde in impulsoen_x:
            tk = float(donde)
            xk = float(x_t(tk))
            graf2[0].stem(tk,xk,
                         linefmt='blue')
    # Graficar terminos con impulso
    impulsoen_h = busca_impulso(ht)
    if len(impulsoen_h)>0:
        for donde in impulsoen_h:
            tk = float(donde)
            hk = float(h_t(tk))
            graf2[0].stem(tk,hk,
                         linefmt='magenta')


    graf2[1].plot(ti,yi,colorlinea_y,
                  label = y_nombre+'(t)')
    # Graficar terminos con impulso
    impulsoen_y = busca_impulso(yt)
    if len(impulsoen_y)>0:
        for donde in impulsoen_y:
            tk = float(donde)
            yk = float(y_t(tk))
            graf2[1].stem(tk,yk,
                         linefmt=colorlinea_y)
    graf2[1].set_xlabel('t')
    graf2[1].legend()
    graf2[1].grid()
    #plt.show()
    return(fig_xh_y)

# GRAFICA CON ANIMACION ------------
def graf_animada_xh_y(xt,ht,yt,t_a,t_b,
                      muestras=101,y_nombre='y',
                      reprod_x = 4,retardo  = 200,
                      archivo_nombre = ''):
    '''grafica animada convolucionx(t) y h(t)
       en dos subgráficas con Parametros de animación trama/foto
        y_nombre = 'ZSR' # o y nombre de resultado convolución
        reprod_x = 4     # velocidad de reproducción
        retardo  = 200   # milisegundos entre tramas
        archivo_nombre = '' # crea gif animado si hay nombre
    '''
    # grafica evaluación numerica
    x_t = sym.lambdify(t,xt,modules=equivalentes)
    h_t = sym.lambdify(t,ht,modules=equivalentes)
    y_t = sym.lambdify(t,yt,modules=equivalentes)

    ti = np.linspace(t_a,t_b,muestras)
    xi = x_t(ti)
    hi = h_t(ti)
    yi = y_t(ti)

    import matplotlib.animation as animation
    # h(t-tau) para cada t
    ht_tau   = []
    for tau in range(0,muestras,reprod_x):
        ht_tau.append(h_t(ti[tau]-ti))
    tramas = len(ht_tau) # tramas creadas

    # figura con dos sub-graficas
    fig_anim = plt.figure()
    graf_a1  = fig_anim.add_subplot(211)
    graf_a2  = fig_anim.add_subplot(212)

    # grafico superior
    x_linea, = graf_a1.plot(ti,xi, color='blue',
                            label=r'$x(\tau)$')
    h_linea, = graf_a1.plot(ti,hi,color='magenta',
                             linestyle='dashed',
                             label=r'$h(\tau)$')
    htau_linea, = graf_a1.plot(ti,ht_tau[0],
                               color='magenta',
                               label=r'$h(t-\tau)$')
    punto1, = graf_a1.plot(0,0, color='magenta',marker=6)

    # grafico inferior
    color_y = 'green'
    if y_nombre=='ZSR':
        color_y ='dodgerblue'
    y_linea, = graf_a2.plot(ti,yi, color=color_y,
                            label=y_nombre+'(t)')
    punto2,  = graf_a2.plot(0,0, color=color_y,marker=6)
    y_sombra, = graf_a2.plot(ti,yi, color=color_y)
    y_sombra.set_visible(False) # Para fijar leyend()


    # Configura gráfica
    titulo = r''+y_nombre+'(t)= x(t)$\circledast$h(t)'
    graf_a1.set_title(titulo)
    ymax1 = np.max([np.max(xi),np.max(hi)])*1.11
    ymin1 = np.min([np.min(xi),np.min(hi)])-0.1*ymax1
    # si hay parte negativa
    if ymin1<0:
        factor = 0.12
        rango = ymax1-ymin1
        ymax1 = ymax1/1.11 + factor*rango
    graf_a1.set_xlim([t_a,t_b])
    graf_a1.set_ylim([ymin1,ymax1])
    graf_a1.set_xlabel(r'$\tau$')
    graf_a1.legend()
    graf_a1.grid()

    ymax2 = np.max(yi)*(1+factor)
    ymin2 = np.min(yi)-0.1*ymax2
    graf_a2.set_xlim([t_a,t_b])
    graf_a2.set_ylim([ymin2,ymax2])
    graf_a2.set_xlabel('t')
    graf_a2.legend()
    graf_a2.grid()

    # cuadros de texto en gráfico
    txt_x = (t_b+t_a)/2
    txt_y = ymax1*(1-(factor-0.01))
    txt_tau = graf_a1.text(txt_x,txt_y,'t='+str(t_a),
                   horizontalalignment='center')

    def trama_actualizar(i,ti,ht_tau):
        # actualiza cada linea
        htau_linea.set_xdata(ti)
        htau_linea.set_ydata(ht_tau[i])

        hasta   = i*reprod_x
        porusar = (muestras-reprod_x*(i+1))
        if porusar>=reprod_x: # en intervalo
            y_linea.set_xdata(ti[0:hasta])
            y_linea.set_ydata(yi[0:hasta])
            punto1.set_xdata(ti[hasta])
            punto1.set_ydata(0)
            punto2.set_xdata(ti[hasta])
            punto2.set_ydata(0)
        else: # insuficientes en intervalo
            y_linea.set_xdata(ti)
            y_linea.set_ydata(yi)
            punto1.set_xdata(ti[-1])
            punto1.set_ydata(0)
            punto2.set_xdata(ti[-1])
            punto2.set_ydata(0)

        # actualiza texto
        t_trama = np.around(ti[i*reprod_x],4)
        txt_tau.set_text('t= '+str(t_trama))
        
        return(htau_linea,y_linea,punto1,punto2,txt_tau)

    def trama_limpiar(): # Limpia Trama anterior
        htau_linea.set_ydata(np.ma.array(ti, mask=True))
        y_linea.set_ydata(np.ma.array(ti, mask=True))
        punto1.set_ydata(np.ma.array(ti, mask=True))
        punto2.set_ydata(np.ma.array(ti, mask=True))
        txt_tau.set_text('')
        return(htau_linea,y_linea,punto1,punto2,txt_tau)

    i   = np.arange(0,tramas,1) # Trama contador
    ani = animation.FuncAnimation(fig_anim,trama_actualizar,i ,
                                  fargs = (ti,ht_tau),
                                  init_func = trama_limpiar,
                                  interval = retardo,
                                  blit=True)
    # Guarda archivo GIF animado o video
    if len(archivo_nombre)>0:
        ani.save(archivo_nombre+'_animado.gif',
                 writer='imagemagick')
        #ani.save(archivo_nombre+'_video.mp4')
    plt.draw()
    #plt.show()
    return(ani)

# Funciones MATG1052.py
def rungekutta2(d1y,x0,y0,h,muestras):
    # Runge Kutta de 2do orden
    tamano = muestras + 1
    estimado = np.zeros(shape=(tamano,2),dtype=float)
    
    # incluye el punto [x0,y0]
    estimado[0] = [x0,y0]
    xi = x0
    yi = y0
    for i in range(1,tamano,1):
        K1 = h * d1y(xi,yi)
        K2 = h * d1y(xi+h, yi + K1)

        yi = yi + (1/2)*(K1+K2)
        xi = xi + h
        
        estimado[i] = [xi,yi]
    return(estimado)

def rungekutta4(d1y,x0,y0,h,muestras):
    # Runge Kutta de 4do orden
    tamano = muestras + 1
    estimado = np.zeros(shape=(tamano,2),dtype=float)
    
    # incluye el punto [x0,y0]
    estimado[0] = [x0,y0]
    xi = x0
    yi = y0
    for i in range(1,tamano,1):
        K1 = h * d1y(xi,yi)
        K2 = h * d1y(xi+h/2, yi + K1/2)
        K3 = h * d1y(xi+h/2, yi + K2/2)
        K4 = h * d1y(xi+h, yi + K3)

        yi = yi + (1/6)*(K1+2*K2+2*K3 +K4)
        xi = xi + h
        
        estimado[i] = [xi,yi]
    return(estimado)

def rungekutta2_fg(f,g,x0,y0,z0,h,muestras):
    tamano = muestras + 1
    estimado = np.zeros(shape=(tamano,3),dtype=float)

    # incluye el punto [x0,y0,z0]
    estimado[0] = [x0,y0,z0]
    xi = x0
    yi = y0
    zi = z0
    for i in range(1,tamano,1):
        K1y = h * f(xi,yi,zi)
        K1z = h * g(xi,yi,zi)
        
        K2y = h * f(xi+h, yi + K1y, zi + K1z)
        K2z = h * g(xi+h, yi + K1y, zi + K1z)

        yi = yi + (K1y+K2y)/2
        zi = zi + (K1z+K2z)/2
        xi = xi + h
        
        estimado[i] = [xi,yi,zi]
    return(estimado)

def rungekutta4_fg(fx,gx,x0,y0,z0,h,muestras):
    tamano = muestras + 1
    estimado = np.zeros(shape=(tamano,3),dtype=float)

    # incluye el punto [x0,y0]
    estimado[0] = [x0,y0,z0]
    xi = x0
    yi = y0
    zi = z0
    
    for i in range(1,tamano,1):
        K1y = h * fx(xi,yi,zi)
        K1z = h * gx(xi,yi,zi)
        
        K2y = h * fx(xi+h/2, yi + K1y/2, zi + K1z/2)
        K2z = h * gx(xi+h/2, yi + K1y/2, zi + K1z/2)
        
        K3y = h * fx(xi+h/2, yi + K2y/2, zi + K2z/2)
        K3z = h * gx(xi+h/2, yi + K2y/2, zi + K2z/2)

        K4y = h * fx(xi+h, yi + K3y, zi + K3z)
        K4z = h * gx(xi+h, yi + K3y, zi + K3z)

        yi = yi + (K1y+2*K2y+2*K3y+K4y)/6
        zi = zi + (K1z+2*K2z+2*K3z+K4z)/6
        xi = xi + h
        
        estimado[i] = [xi,yi,zi]
    return(estimado)

def edo_lineal_auxiliar(ecuacion,
                 t = sym.Symbol('t'),r = sym.Symbol('r'),
                 y = sym.Function('y'),x = sym.Function('x')):
    ''' ecuacion auxiliar o caracteristica de EDO
        t independiente
    '''
    # ecuación homogénea x(t)=0, entrada cero
    RHSx0 = ecuacion.rhs.subs(x(t),0).doit()
    LHSx0 = ecuacion.lhs.subs(x(t),0).doit()
    homogenea = LHSx0 - RHSx0
    homogenea = sym.expand(homogenea,t)

    # ecuación auxiliar o característica
    Q = 0*r
    term_suma = sym.Add.make_args(homogenea)
    for term_k in term_suma:
        orden_k = sym.ode_order(term_k,y)
        coef = 1 # coefientes del término suma
        factor_mul = sym.Mul.make_args(term_k)
        for factor_k in factor_mul:
            cond = factor_k.has(sym.Derivative)
            cond = cond or factor_k.has(y(t))
            if not(cond):
                coef = coef*factor_k
        Q = Q + coef*(r**orden_k)
               
    # Q factores y raices
    Q_factor = sym.factor(Q,r)
    Q_poly   = sym.poly(Q,r)
    Q_raiz   = sym.roots(Q_poly)
    
    auxiliar = {'homogenea' : sym.Eq(homogenea,0),
                'auxiliar'  : Q,
                'Q'         : Q,
                'Q_factor'  : Q_factor,
                'Q_raiz'    : Q_raiz }
    return(auxiliar)

def edo_lineal_complemento(ecuacion,y_cond):
    # ecuación homogénea x(t)=0, entrada cero
    RHSx0 = ecuacion.rhs.subs(x(t),0).doit()
    LHSx0 = ecuacion.lhs.subs(x(t),0).doit()
    homogenea = LHSx0 - RHSx0

    # solucion general de ecuación homogénea
    general = sym.dsolve(homogenea, y(t))
    general = general.expand()

    # Aplica condiciones iniciales o de frontera
    eq_condicion = []
    for cond_k in y_cond: # cada condición
        valor_k = y_cond[cond_k]
        orden_k = sym.ode_order(cond_k,y)
        if orden_k==0: # condicion frontera
            t_k    = cond_k.args[0] # f(t_k)
            expr_k = general.rhs.subs(t,t_k)
        else: # orden_k>0
            subs_param = cond_k.args[2] # en valores
            t_k = subs_param.args[0]  # primer valor
            dyk = sym.diff(general.rhs,t,orden_k)
            expr_k = dyk.subs(t,t_k)
        eq_condicion.append(sym.Eq(valor_k,expr_k))

    constante = sym.solve(eq_condicion)

    # ecuacion complementaria
    # reemplaza las constantes en general
    y_c = general
    for Ci in constante:
        y_c = y_c.subs(Ci, constante[Ci])
    
    complemento = {'homogenea'      : sym.Eq(homogenea,0),
                   'general'        : general,
                   'eq_condicion'   : eq_condicion,
                   'constante'      : constante,
                   'complementaria' : y_c}
    return(complemento)

def edo_lineal_particular(ecuacion,xp):
    ''' edo solucion particular con entrada x(t)
    '''
    # ecuación particular x(t)=0, estado cero
    RHSxp = ecuacion.rhs.subs(x(t),xp).doit()
    LHSxp = ecuacion.lhs.subs(x(t),xp).doit()
    particular = LHSxp - RHSxp

    # solucion particular de ecuación homogénea
    yp = sym.dsolve(particular, y(t))

    # particular sin terminos Ci
    y_Ci = yp.free_symbols
    y_Ci.remove(t) # solo Ci
    for Ci in y_Ci: 
        yp = yp.subs(Ci,0)
        
    # simplifica y(t) y agrupa por escalon unitario
    yp = sym.expand(yp.rhs,t)
    lista_escalon = yp.atoms(sym.Heaviside)
    yp = sym.collect(yp,lista_escalon)
    yp = sym.Eq(y(t),yp)
    
    return(yp)

def print_resultado_dict(resultado):
    ''' print de diccionario resultado
        formato de pantalla
    '''
    eq_sistema = ['ZIR','h','ZSR','xh',
                  'H','Hs','Hs_fp','yt_ZIR',
                  'yt_ZSR']
    for entrada in resultado:
        tipo = type(resultado[entrada])
        cond = (tipo == sym.core.relational.Equality)
        cond = cond or (entrada in eq_sistema)
        if cond:
            print(entrada,':')
            sym.pprint(resultado[entrada])
        elif tipo==list or tipo==tuple:
            tipoelem = type(resultado[entrada][0])
            if tipoelem == sym.core.relational.Equality:
                print(entrada,':')
                for fila in resultado[entrada]:
                    sym.pprint(fila)
            elif tipoelem == list:
                print(entrada,':')
                for fila in resultado[entrada]:
                    print(' ',fila)
            elif tipoelem == tuple:
                print(entrada,':')
                for fila in resultado[entrada]:
                    print(' ',fila)
            else:
                print(entrada,':',resultado[entrada])
        elif tipo==dict:
            if (entrada=='ZIR_Qs2' or entrada=='ZSR_Qs2'):
                print('\n',entrada,':')
                for fila in resultado[entrada]:
                    print('',fila,':')
                    print(' ',resultado[entrada][fila])
            else:
                print(entrada,':',resultado[entrada])
        else:
            print(entrada,':',resultado[entrada])
    return()

# UNIDAD 5
# LTI CT Transformadas de Fourier
# http://blog.espol.edu.ec/telg1001/series-de-fourier-senales-periodicas-ejemplos-con-python/

def fourier_series_coef(serieF,n,T0, casicero=1e-10):
    ''' coeficientes de serie de Fourier
        ak,bk,ck_mag, ck_fase
    '''
    w0 = 2*sym.pi/T0
    ak = [float(serieF.a0.evalf())]
    bk = [0]; k_i=[0]
    ak_coef = serieF.an
    bk_coef = serieF.bn
    ck_mag  = [ak[0]] ; ck_fase = [0]
    for i in range(1,n,1):
        ak_valor = ak_coef.coeff(i).subs(t,0)
        ak_valor = float(ak_valor.evalf())
        ak.append(ak_valor)
        bk_term = bk_coef.coeff(i).evalf()
        bk_valor = 0
        term_mul = sym.Mul.make_args(bk_term)
        for term_k in term_mul:
            if not(term_k.has(sym.sin)):
                bk_valor = float(term_k)
            else: # sin(2*w0*t)
                ki = term_k.args[0].subs(t,1)/w0
        bk.append(bk_valor)
        k_i.append(i)
        
        # magnitud y fase
        ak_signo = 1 ; bk_signo = 1
        if abs(ak_valor)>casicero:
            ak_signo = np.sign(ak_valor)
        if abs(bk_valor)>casicero:
            bk_signo = np.sign(bk_valor)
        signo_ck = ak_signo*bk_signo
        ck_mvalor = signo_ck*np.sqrt(ak_valor**2 + bk_valor**2)
        ck_mag.append(ck_mvalor)
        pendiente = np.nan
        if (abs(ak_valor)>=casicero):
            pendiente = -bk_valor/ak_valor
        ck_fvalor = np.arctan(pendiente)
        ck_fase.append(ck_fvalor)
    coef_fourier = {'k_i': k_i,'ak': ak,'bk': bk,
                    'ck_mag' : ck_mag,'ck_fase': ck_fase}
    return (coef_fourier)

def graficar_ft_periodoT0(ft,t0,T0,muestras=51,
                          n_periodos=4,f_nombre='f'):
    ''' grafica f(t) en intervalo[t0,t0+T0] para n_periodos
    '''
    # convierte a sympy una constante
    ft = sym.sympify(ft,t) 
    if ft.has(t): # no es constante
        f_t = sym.lambdify(t,ft,modules=equivalentes)
    else:
        f_t = lambda t: ft + 0*t
    # intervalo de n_periodos
    ti = np.linspace(float(t0-T0*n_periodos/2),
                     float(t0+T0*n_periodos/2),
                     n_periodos*muestras)
    fk = np.zeros(n_periodos*muestras)
    # ajuste de intervalo por periodos
    ti_T0 = (ti-t0)%float(T0)+t0
    fi = f_t(ti_T0)
    
    # intervalo de UN periodo
    ti0 = np.linspace(float(t0),float(t0+T0),muestras)
    fi0 = f_t(ti0)
    
    fig_fT0, graf_fT0 = plt.subplots()
    graf_fT0.plot(ti,fi,label=f_nombre+'(t)',
                 color= 'blue',linestyle='dashed')
    graf_fT0.plot(ti0,fi0,label=f_nombre+'(t)',color= 'blue')
    graf_fT0.axvline(t0, color ='red')
    graf_fT0.axvline(t0+T0, color ='red')
    graf_fT0.set_xlabel('t')
    graf_fT0.set_ylabel(f_nombre+'(t)')
    graf_fT0.legend()
    graf_fT0.grid()
    ft_etq = ''
    if not(ft.has(sym.Piecewise)):
        ft_etq = '$ = '+str(sym.latex(ft)) +'$'
    etiq_1 = r''+f_nombre+'(t) '+ft_etq+' ; $T_0='+str(sym.latex(T0))+'$'
    graf_fT0.set_title(etiq_1)
    return(fig_fT0)

def graficar_w_espectro(coef_fourier,T0,f_nombre='f'):
    ''' coef_fourier es diccionario con entradas
        ['k_i','ck_mag','ck_fase'] indice, Ck_magnitud, Ck_fase
    '''
    # espectro de frecuencia
    k_i = coef_fourier['k_i']
    ck  = coef_fourier['ck_mag']
    cfs = coef_fourier['ck_fase']

    # grafica de espectro de frecuencia
    fig_espectro_w, graf_spctr = plt.subplots(2,1)
    graf_spctr[0].stem(k_i,ck,label='Ck_magnitud')
    graf_spctr[0].set_ylabel('|Ck|')
    graf_spctr[0].legend()
    graf_spctr[0].grid()
    ft_etq = ''
    if not(ft.has(sym.Piecewise)):
        ft_etq = '$ = '+str(sym.latex(ft)) +'$'
    etiq_2 = ft_etq+' ; $T_0='+str(sym.latex(T0))+'$'
    etiq_1 = r'Espectro frecuencia '+f_nombre+'(t) '
    graf_spctr[0].set_title(etiq_1+etiq_2)
    graf_spctr[1].stem(k_i,cfs,label='Ck_fase')
    graf_spctr[1].legend()
    graf_spctr[1].set_ylabel('Ck_fase')
    graf_spctr[1].set_xlabel('k')
    graf_spctr[1].grid()
    return(fig_espectro_w)

# Unidad 7
def apart_z(Fz):
    ''' fracciones parciales en dominio z
        modifica con factor 1/z
    '''
    Fz = sym.simplify(Fz)
    # fracciones parciales modificadas con 1/z
    Fzz = (Fz)/z
    Fzm = sym.apart(Fzz,z)
    # restaura z
    term_suma = sym.Add.make_args(Fzm)
    Fzp = 0*z
    for term_k in term_suma:
        Fzp = Fzp + term_k*z
    return(Fzp)

def Q_cuad__zparametros(Fz):
    ''' parametros cuadraticos en dominio z
    '''

    def Q_cuad_z_term(untermino):
        ''' parametros cuadraticos en dominio z
            de un termino de fraccin parcial
        '''
        unparametro ={}
        # revisa denominador cuadratico
        [numerador,denominador] = (untermino).as_numer_denom()
        gradoD = 0
        coeficientesD = denominador
        gradoN = 0
        coeficientesN = numerador
        if not(denominador.is_constant()):
            denominador = denominador.as_poly()
            gradoD = denominador.degree()
            coeficientesD = denominador.coeffs()
        if not(numerador.is_constant()):
            numerador = numerador.as_poly()
            gradoN = numerador.degree()
            coeficientesN = numerador.coeffs()
        if gradoD == 2 and gradoN==2:
            a = float(coeficientesD[1])/2
            gamma2 = float(coeficientesD[2])
            gamma = np.sqrt(gamma2)
            A = float(coeficientesN[0])
            B = float(coeficientesN[1])
            rN = (A**2)*gamma2 + B**2 - 2*A*a*B
            rD = gamma2 - a**2
            r = np.sqrt(rN/rD)
            beta = np.arccos(-a/gamma)
            thetaN = A*a-B
            thetaD = A*np.sqrt(gamma2-a**2)
            theta = np.arctan(thetaN/thetaD)
            unparametro = {'r':r,
                           'gamma':gamma,
                           'beta':beta,
                           'theta':theta}
        return(unparametro)

    Fz = apart_z(Fz)
    # parametros denominador cuadratico
    Qs2 = {}
    term_suma = sym.Add.make_args(Fz)
    for term_k in term_suma:
        Qs2_k = Q_cuad_z_term(term_k)
        if len(Qs2_k)>0:
            Qs2[term_k] = Qs2_k
    return(Qs2)

def graficar_Fz_polos(Fz,Q_polos={},P_ceros={},
                z_a=1,z_b=0,muestras=101,f_nombre='F',
                solopolos=False,precision=4,casicero=1e-10):
    ''' polos y ceros plano z imaginario
    '''
    fig_zROC, graf_ROC = plt.subplots()
    # limite con radio 1
    radio1 = plt.Circle((0,0),1,color='lightsalmon',
                        fill=True)
    radio2 = plt.Circle((0,0),1,linestyle='dashed',
                        color='orange',fill=False)
    graf_ROC.add_patch(radio1)
    for unpolo in Q_polos.keys():
        [r_real,r_imag] = unpolo.as_real_imag()
        unpolo_radio = np.abs(unpolo)
        unpolo_ROC = plt.Circle((0,0),unpolo_radio,
                          color='lightgreen',fill=True)
        graf_ROC.add_patch(unpolo_ROC)
    graf_ROC.add_patch(radio2) # borde r=1
    graf_ROC.axis('equal')
    # marcas de r=1 y polos
    for unpolo in Q_polos.keys():
        x_polo = np.round(float(sym.re(unpolo)),precision)
        y_polo = np.round(float(sym.im(unpolo)),precision)
        etiq_polo = x_polo
        if y_polo>casicero:
            etiq_polo= etiq_plo+1j*y_polo
        etiqueta = 'polo: '+str(etiq_polo)
        graf_ROC.scatter(x_polo,y_polo,marker='x',
                        color='red',label = etiqueta)
        etiqueta = "("+str(x_polo) +','+str(y_polo)+")"
        plt.annotate(etiqueta,(x_polo,y_polo), rotation=45)
    # marcas de ceros
    for uncero in P_ceros.keys():
        x_cero = np.round(float(sym.re(uncero)),precision)
        y_cero = np.round(float(sym.im(uncero)),precision)
        etiq_cero = x_cero
        if y_cero>casicero:
            etiq_cero= etiq_plo+1j*y_cero
        etiqueta = 'cero: '+str(etiq_cero)
        graf_ROC.scatter(x_cero,y_cero,marker='o',
                        color='blue',label = etiqueta)
        etiqueta = "("+str(x_cero) + ','+str(y_cero)+")"
        plt.annotate(etiqueta,(x_cero,y_cero), rotation=45)
    # limita radio 1
    graf_ROC.plot(1,0,'o',color='red',
                 label ='radio:'+str(1))
    graf_ROC.axhline(0,color='grey')
    graf_ROC.axvline(0,color='grey')
    graf_ROC.grid()
    graf_ROC.legend()
    graf_ROC.set_xlabel('Re[z]')
    graf_ROC.set_ylabel('Imag[z]')
    untitulo = r'ROC '+f_nombre+'[z]=$'
    untitulo = untitulo+str(sym.latex(Fz))+'$'
    graf_ROC.set_title(untitulo)
    return(fig_zROC)
