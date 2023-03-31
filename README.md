# vehicle_classfication

Lo que vamos a tratar de hacer en esta practica es entrenar una red neuronal convolucional con el fin de que al final del entrenamiento seaz capaz de identificar distintos vehiculos de transporte.

Partimos de una base que no es la optima. Los datos que tenemos son un conjunto de fotos de los distintos transportes circulando por las autovias. Para realizar una lectura de las imágenes es necesario el uso de librerías de python como Beautiful soup, que puede leer páginas HTML para poder leer fotos subidas a la nube.

Como no podemos hacer una buena lectura de los datos dados ni como es obvio su posterior entrenamiento de forma optima, hemos tratado de modificar el codigo dado en la anterior practica en la que la red neuronal diferenciaba entre perros y gatos. Al no ser fotos enteramente de los vehiculos en sí mismos, sino que hay que leerlos con sus respectivos labels hemos obtenido diferentes errores en la lectura de las imagenes.

Lo que se entrega es una implmentación y adaptación del código, obteniendo una red neuronal que en lugar de clasificar entre dos objetos distintos, clasifica entre distintos vehículos.