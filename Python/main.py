import math
import random 
from gui import GUI


#dataset 
dataset=[
[1.75,7.0,-1],
[1.54,4.5,1],
[1.60,5.1,1],
[1.76,7.5,-1],
[1.55,4.0,1],
[1.81,8.0,-1],
[1.80,8.2,-1],
[1.56,4.3,1],
[1.68,7.1,-1],
[1.60,5.5,1],
[1.70,6.5,-1],
[1.55,5.9,1],
[1.80,7.0,-1],
[1.70,6.0,-1],
[1.80,7.0,1],
[1.60,5.0,1],
[1.55,5.0,1]]    

#dataset da testare
toAnalize=[
[1.75,7.5],
[1.54,5.2],
[1.59, 8.0],
[1.76,7.0],
[1.55,4.0],
[1.81,8.0],
[1.80,8.2],
[1.56,4.3],
[1.68,7.1],
[1.80,7.0],]  


def main():
    random.seed(1)
    #Inizializzazione dei pesi in maniera casuale.
    #Il peso e' quel numero che viene moltiplicato per il valore
    #del neurone di partenza per arrivare a quello di output.
    w1 = random.random()
    w2 = random.random()
    b = random.random()
    #valori dei pesi "corretti" 
    w1, w2, b = train(w1, w2, b)
    print(w1,w2,b)
    pred=[] 
    for person in toAnalize:  
        z = w1 * person[0] + w2 * person[1] + b
        prediction=app(tanh(z))
        
        
        answer = neuron_man(prediction)
        print(answer)
        if answer == "":
            answer = neuron_woman(prediction)
        pred.append(answer)
        
    print(pred)
    #GUI.inizializePlot(dataset,toAnalize, w1, w2, b)
    #GUI.showPlot

#Una funzione sigmoide e' quella famiglia di funzioni con la 
#particolare caratteristica di essere s-shaped.
#Ha come dominio tutto R e, in questo caso particolare, ha come
#codominio l'intervallo [0,1].
#Questa particolare funzione viene chiamata Standard Logistic Function.
#Altre funzioni sigmoide sono:
#1) Tangente hyperbolica
#2) Arcotangeante

def sigmoide(t):
    return 1/(1+math.exp(-t))

def tanh(t):
    return  (math.exp(t) - math.exp(-t))/(math.exp(t) + math.exp(-t))

def tanh_p(t):
    return 1 - tanh(t)**2

#Derivata prima della funzione sigmoide
#Particolarita' della funzione:
#1) Bell-shaped, come per tutte le altri funzioni sigmoide
#2) Viene chiamata Logistic distribution
#3) Negli scacchi si e' passati dall'uso della distribuzione normale(Gaussian distribution)
# a quella logistica.
#4) y' = y*(1-y)
def sigmoide_p(t):
    return sigmoide(t)*(1 - sigmoide(t))

def train(w1, w2 ,b):

    #numero di cicli di apprendimento
    iteration = 100

    #intero che indica la veloci.ta' di apprendimento
    #indica di quanto il valore dei pesi deve essere modificato
    #rispetto al valore obiettivo.
    #Se volessi far abbassare la pendenza della funzione di costo, per far
    #tendere lo scarto quadratico a zero (x-obiettivo)^2.
    #dato un valore x, se il mio obiettivo e' 2, la mia x deve avvicinarsi a 2.
    #tramite la derivata, se il mio x e' > 2, la derivata sara' positiva quindi devo diminuirlo
    #se il mio x e' < 2, la derivata sara' negativa quindi devo aumentarlo.
    #Il learning rate indica quanto devo aumentare o diminuire il valore di x.
    #x=x-pendenza. Se pero' non si usa il learning rate, x diventerà -x e così via.
    #far vedere disegnigno.
    learningRate = 0.8
    for i in range(iteration):
        for person in dataset:
            point = person
            z = point[0] * w1 + point[1] * w2 + b
            #supp = tanh(z)
            #print(supp)
            #pred = app(supp) 
            pred = tanh(z)
            target = point[2] 
            #data la funzione di costo c = (sigmoide(point[0]*w1, point[1]*w2, b) - target)^2
            #la sua derivata sara' c' sara' il prodotto di questi elementi 
            # 2*(pred-target) = dcosto/dpred
            # sigmoide_p(z) = dpred/dz
            # point[0] = dz/dw1, in quanto w1 essendo una variabile diventa 1, rimane point[0].
            #point[1]w2 + b possono essere visti come costante quindi la derivata e' zero.
            a = ((2*(pred - target)) * tanh_p(z) * point[0])
            print(learningRate * a, target)
            w1 = w1 - learningRate * a
            w2 = w2 - learningRate * ((2*(pred - target)) * tanh_p(z) * point[1])
            b = b - learningRate * ((2*(pred - target)) * tanh_p(z) * 1)
        
    return w1, w2, b
def app(n):
    if n >= 0:
        return 1
    return -1
def neuron_man(prediction):
    prediction = prediction * (-1)
    if prediction == 1:
        return "man"
    else:
        return ""

def neuron_woman(prediction):
    prediction = prediction * 1
    if prediction == 1:
        return "woman"
    else:
        return ""
main()
