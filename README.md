# vehicle_classfication

Lo que vamos a tratar de hacer en esta practica es entrenar una red neuronal convolucional con el fin de que al final del entrenamiento seaz capaz de identificar distintos vehiculos de transporte.

Como no podemos hacer una buena lectura de los datos dados ni como es obvio su posterior entrenamiento de forma optima, hemos tratado de modificar el codigo dado en la anterior practica en la que la red neuronal diferenciaba entre perros y gatos. Al no ser fotos enteramente de los vehiculos en sí mismos, sino que hay que leerlos con sus respectivos labels hemos obtenido diferentes errores en la lectura de las imagenes.

Se entrega una implementación y adaptación del código, obteniendo una red neuronal que en lugar de clasificar entre dos objetos distintos, clasifica entre siete objetos.

En la parte del entrenamiento obtenemos un error que se debe a que ciertas imagenes no se pueden leer de forma efectiva por lo mencionado anteriormente. Si hiciesemos una limpieza en condiciones de los archivos, el codigo funcionaria perfectamente. En la parte de prediccion, se calcula de forma eficiente el numero de aciertos y fallos al leer los archivos. Pasando al tema de dockerizacion, aunque nos ha dado el mismo error que en la anterior practica, lo hemos intentado de nuevo siguiendo todos los pasos, instalando las dependencias y su posterior conexion con una base de datos mongo. Al intentar conectar las imagenes con sus respectivos volumenes nos encontramos con problemas.