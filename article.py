import streamlit as st
from lib.utils import Stats

stats = Stats() 
st.markdown('''
# Los números feos de la lotería

\nExisten pocos tesoros más buscados que un billete de lotería ganador. Sin embargo, en el sorteo de Navidad de 2017, la administración Ormaetxea de Bilbao (tan popular como podría serlo “Doña Manolita” en Madrid) tuvo que devolver dos series completas de billetes agraciados con el tercer y quinto premio. El motivo: al tratarse de “números feos”, nadie había querido comprarlos. Preguntado sobre el asunto, el gerente de la lotería, Sergio Etxebarria, explicaba: «**Nadie lo ha querido, no se han atrevido con él**». «Y esto que nos ha pasado a nosotros habrá pasado en muchas administraciones de España porque los números bajos cuesta mucho venderlos».
\nLos adefesios aquel año fueron el 00750 (tercer premio), 06293 (quinto). Por su culpa, Bilbao sacrificó 360.000 € a su ideal colectivo de belleza numérica. Pero no era la primera vez, ni sería la última, en que sucedía algo parecido. El «Gordo Feo» de 2023 fue el 88008, lleno de agujeros y sospechosas repeticiones. En 2018 la suerte cayó sobre el 3347, con un cero a la izquierda tan inútil como desagradable a la vista. Pero el peor escenario de todos se dio en 2010: el segundo premio quedó desierto, cuando recayó sobre el 147. «Como tenía dos ceros por delante la gente lo descarta», como cuenta Etxebarria. «Los más maniáticos creen incluso que si les ofreces uno tan bajo les quieres engañar».\n  
![Los números feos de la lotería][def]

[def]: https://raw.githubusercontent.com/almudenamcastro/number_beauty/refs/heads/main/resources/image.png

\nLa idea aversión hacia estos "números feos" se ha vuelto tan común que en algunas loterías se han convertido en un fetiche. Y es curioso, desde un punto de vista matemático, **su suerte no difiere en absoluto de la de sus hermanos más guapos**. 

\n¿En qué se basa entonces esta curiosa superstición? Y, sobre todo, ¿qué es lo que hace que un número nos parezca "feo" o más "bonito"?

\nHemos mirado a las ventas de lotería de los últimos años para entender qué se vende más y qué menos. Que es lo que hace que un número nos parezca "bonito", o no queramos tocarlo ni con un palo. Esto es lo que hemos encontrado. 

### Los parámetros de la belleza numérica: 

#### El 13, un número de la buena suerte:
       
''')
st.write("")
col1, col2 = st.columns(2)

with col1:
    st.markdown('**Números acados en 13**')
    st.write('¡Se venden casi todo!')
    st.pyplot(stats.plot('ends_13', 'Termina en 13','x'))
with col2:
    st.markdown('**Números que empiezan en 13**')
    st.write('Se venden en un 90%')
    st.pyplot(stats.plot('starts_13', 'Empieza en 13', 'x'))
