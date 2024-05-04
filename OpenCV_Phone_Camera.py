import cv2
import numpy as np
import requests

# Remplacer par l'adresse IP locale de votre téléphone
ip_address = "http://192.168.191.2:8080/shot.jpg"

while True:
    # Récupérer l'image du flux vidéo
    response = requests.get(ip_address)
    image_data = response.content

    
    # Décoder l'image en format BGR
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

    # Redimensionner l'image (50% de la taille d'origine) Premier Traitement
    image_resized = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

    # Convertir l'image en nuances de gris
    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)


    #Application des filtres


    # Appliquer un flou gaussien (taille du noyau 5x5)
    # image_blurred = cv2.GaussianBlur(image_resized, (5, 5), 0)

    # Détection des contours (algorithme Canny)
    image_edges = cv2.Canny(image_gray, 50, 150)


    # Afficher l'image
    cv2.imshow('Flux vidéo', image_edges)

    # Attendre une touche pour quitter
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cv2.destroyAllWindows()
