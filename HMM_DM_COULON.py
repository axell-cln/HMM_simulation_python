# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 15:28:19 2022

@author: axelc
"""

import numpy as np  
from random import choices

pi0=np.transpose([0.5,0.5,0,0,0,0,0,0]) # DISTRIBUTION INITIALE SUR LES ETATS A T=0
observationsPossibles=['11','10','01','00'] # LE PREMIER INDEX EST L'ETAT BOOLEEN DU CAPTEUR EN S1 ET LE DEUXIEME INDEX EST L'ETAT DU CAPTEUR EN S4.
etats=[0,1,2,3,4,5,6,7]


""" Méthodes qui servent à construire les matrices de notre HMM """

def buildFonctionTransition(etatDepart,etatArrivee):
    """Cas particulier"""
    if(etatDepart==etats[-1] and etatArrivee==etats[0]):
        return 0.7
    if(etatArrivee==etats[-1] and etatDepart==etats[0]):
        return 0.1
    
    """Fonctionnement général"""
    if(abs((etatArrivee-etatDepart))>1):
        return 0
    if(etatArrivee<etatDepart):
        return 0.1
    if(etatDepart<etatArrivee):
        return 0.7
    if(etatArrivee==etatDepart):
        return 0.2
   
    
def buildFonctionObservation(etat,observation):
   doudouSurCapteurMaisPasDetecte=0.1
   doudouSurCapteurEtDetecte=(1-doudouSurCapteurMaisPasDetecte)
   doudouPasSurCapteurMaisDetecte=0.2
   doudouPasSurCapteurEtPasDetecte=(1-doudouPasSurCapteurMaisDetecte)
   caseCapteur1=1
   caseCapteur2=4
   
   if(observation=='11'):
       if(etat!=caseCapteur1 and etat!=caseCapteur2):
           return doudouPasSurCapteurMaisDetecte*doudouPasSurCapteurMaisDetecte
       else:
           return doudouPasSurCapteurMaisDetecte*doudouSurCapteurEtDetecte
           
   if(observation=='10'):
       if(etat!=caseCapteur1 and etat!=caseCapteur2):
           return doudouPasSurCapteurMaisDetecte*doudouPasSurCapteurEtPasDetecte
       elif(etat==caseCapteur1):
           return doudouSurCapteurEtDetecte*doudouPasSurCapteurEtPasDetecte
       elif(etat==caseCapteur2):
           return doudouPasSurCapteurMaisDetecte*doudouSurCapteurMaisPasDetecte
    
   if(observation=='01'):
       if(etat!=caseCapteur1 and etat!=caseCapteur2):
           return doudouPasSurCapteurMaisDetecte*doudouPasSurCapteurEtPasDetecte
       elif(etat==caseCapteur1):
           return doudouPasSurCapteurMaisDetecte*doudouSurCapteurMaisPasDetecte
       elif(etat==caseCapteur2):
           return doudouSurCapteurEtDetecte*doudouPasSurCapteurEtPasDetecte
       
   if(observation=='00'):
       if(etat!=caseCapteur1 and etat!=caseCapteur2):
           return doudouPasSurCapteurEtPasDetecte*doudouPasSurCapteurEtPasDetecte
       else:
           return doudouSurCapteurMaisPasDetecte*doudouPasSurCapteurEtPasDetecte
     
           
       
""" Matrice de Transition """

matriceTransition=np.zeros((len(etats),len(etats)))
for etatArrivee in etats:  
    for etatDepart in etats:
        matriceTransition[etatArrivee,etatDepart]=buildFonctionTransition(etatDepart,etatArrivee)
print("Matrice de transtion: ",matriceTransition)



""" Matrice d'observations """

i=0
j=0
matriceObservations=np.zeros((len(observationsPossibles),len(etats)))
for observation in observationsPossibles:      
    for etat in etats:
        matriceObservations[i,j]=buildFonctionObservation(etat,observation)
        j+=1
    j=0
    i+=1
        
print("Matrice d'observations: ",matriceObservations)
    


""" Classe pour construire notre HMM """

class HMM:
    
    def __init__(self,etats,matriceTransition,observationsPossibles,matriceObservation,distribInitiale):
        self.etats = etats
        self.matriceTransition = matriceTransition
        self.observationsPossibles = observationsPossibles
        self.matriceObservation = matriceObservation
        self.distribInitiale = distribInitiale
        
    def etats(self):
        return self.etats
    
    def observations(self):
        return self.observationsPossibles
  
    def prediction(self,st):
        stPlus1=self.matriceTransition@st # MULTIPLICATION MATRICIELLE
        return stPlus1
    
    def correction(self,stPlus1,observationTPlus1):
        if(observationTPlus1==self.observationsPossibles[0]):
            pObservationT1SachantEtatT1=self.matriceObservation[0,:]
        if(observationTPlus1==self.observationsPossibles[1]):
            pObservationT1SachantEtatT1=self.matriceObservation[1,:]
        if(observationTPlus1==self.observationsPossibles[2]):
            pObservationT1SachantEtatT1=self.matriceObservation[2,:]
        if(observationTPlus1==self.observationsPossibles[3]):
            pObservationT1SachantEtatT1=self.matriceObservation[3,:]
        pObservationTPlus1=sum(pObservationT1SachantEtatT1*stPlus1)
        pEtatT1SachantObservationT1=pObservationT1SachantEtatT1*stPlus1/pObservationTPlus1
        return pEtatT1SachantObservationT1
    
    def propagation(self,st,observationTPlus1):
        stPlus1=self.prediction(st)
        pEtatT1SachantObservationT1=self.correction(stPlus1,observationTPlus1)
        return pEtatT1SachantObservationT1
    
    
    def filtrage(self,st,observations : []):
        for n in range(len(observations)) :
            stn=self.propagation(st,observations[n])
            st=stn
        return stn
    
    """Algorithme de Viterbi"""
    
    """Suite d'observations : 1,0 0,1 1,0 0,0"""
    def viterbi(self, liste_observations, distribInitiale):
        meilleurParcours = np.zeros((liste_observations.shape[0]+1, len(distribInitiale)))
        meilleurParcours[0] = distribInitiale
        ligne = 0
        for observation in liste_observations:
            ligne += 1
            for colonne in range(len(distribInitiale)):
                for indice in range(len(distribInitiale)):
                    meilleurParcours[ligne, colonne] += meilleurParcours[ligne - 1, indice]*buildFonctionTransition(self.etats[indice], self.etats[colonne])
            for colonne in range(len(distribInitiale)):
                meilleurParcours[ligne, colonne] *= buildFonctionObservation(self.etats[colonne], observation)
        lienASuivre = np.argmax(meilleurParcours, axis=1)
        return lienASuivre
     


"""-----------------------------------------------------------------------------------PARTIE 2--------------------------------------------------------------------------------"""

HMMDoudou=HMM(etats,matriceTransition,observationsPossibles,matriceObservations,pi0)

"""Validation prédiction avec l'état initial"""
predictionS1=HMMDoudou.prediction(HMMDoudou.distribInitiale)
print("(2c) Validation prédiction: ",predictionS1)

"""Validation correction"""
correctionS1=HMMDoudou.correction(predictionS1,'10')
for i in range(len(correctionS1)):
    correctionS1[i]=round(correctionS1[i],3)
print("(2e) Validation correction: ",correctionS1)

"""Validation Filtrage"""
suite_observations=['10','01','10']
st3=HMMDoudou.filtrage(HMMDoudou.distribInitiale,suite_observations)
for i in range(len(st3)):
    st3[i]=round(st3[i],3)
print("(2h) Validation Filtrage: ",st3)
   

 
class Systeme:
    
    observation=""
    sens_rotation=[]
    def __init__(self,HMM,distributionInitiale):
        self.HMM=HMM
        self.distributionCourante=distributionInitiale
        self.etatCourant=choices(self.HMM.etats,distributionInitiale)
        
    def evoluerEtatReel(self):
        nouveauEtatCourant=choices(self.HMM.etats,self.HMM.matriceTransition[:,self.etatCourant])
        self.etatCourant=nouveauEtatCourant
        self.observation=choices(self.HMM.observations(),self.HMM.matriceObservation[:,self.etatCourant])
        return
    
    def mettreAJourBelief(self):
        self.distributionCourante=self.HMM.propagation(self.distributionCourante,self.observation[0])
        return
    
    def evoluerSysteme(self):
        self.evoluerEtatReel()
        self.mettreAJourBelief()
        return
    def mainSysteme(self):
        print("Etat initial avant evolution: ",self.etatCourant)
        for i in range(5):
            self.evoluerSysteme()
            #self.sens_rotation.append(self.observation[0])
            print("Distribution Courante: ",self.distributionCourante)
            print("Observation Générée: ",self.observation[0])
            print("Nouveau Etat courant: ",self.etatCourant)
            #print("Suivi du sens de rotation :",  self.sens_rotation)
        return
    def sensRotation(self):
        for i in range(50):
            self.evoluerSysteme()
            index_max=np.argmax(self.distributionCourante)
            if index_max==1:
                obs_parfaite='10'
            elif index_max==4:
                obs_parfaite='01'
            else:
                obs_parfaite='00'
            self.sens_rotation.append(obs_parfaite)
            l=len(self.sens_rotation)-1
            if(l>3):
                if (self.sens_rotation[l]=='01' and self.sens_rotation[l-1]=='00' and self.sens_rotation[l-2]=='00' and self.sens_rotation[l-3]=='10' ):
                    print("Sens Horaire ! ",self.sens_rotation)
                    break 
                elif (self.sens_rotation[l]=='10' and self.sens_rotation[l-1]=='00' and self.sens_rotation[l-2]=='00' and self.sens_rotation[l-3]=='01'  and self.sens_rotation[l-4]=='00'  and self.sens_rotation[l-5]=='00'):
                    print("Sens Anti Horaire ! ",self.sens_rotation)
                    break 
        
HMMDoudou_reset=HMM(etats,matriceTransition,observationsPossibles,matriceObservations,pi0)
systemeDoudou=Systeme(HMMDoudou_reset,pi0)
print("(3e) Evolution du système: ")
systemeDoudou.mainSysteme()

HMMDoudou_sens_rota=HMM(etats,matriceTransition,observationsPossibles,matriceObservations,pi0)
systemeDoudou_sens_rota=Systeme(HMMDoudou_sens_rota,pi0)
print("(4a) Sens de rotation: ")
systemeDoudou_sens_rota.sensRotation()

commenceASo = np.array([1, 0, 0, 0, 0, 0, 0, 0])
HMMDoudou_viterbi=HMM(etats,matriceTransition,observationsPossibles,matriceObservations,commenceASo)
liste_observations = np.array([['10'], ['01'], ['10'], ['00']])
print("(5a)Viterbi pour la séquence: ",liste_observations)
print("Séquence d'états: ",HMMDoudou_viterbi.viterbi(liste_observations,HMMDoudou_viterbi.distribInitiale))

