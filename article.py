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

''')            
st.markdown('''
### Los parámetros de la belleza numérica: 

#### - El 13, un número de la buena suerte:
Que un número termine en 13 es, prácticamente, garantía de venta. En promedio se venden el 98,77% de los billetes con esta terminación. 
Al inicio del número tiene un poco menos de impacto, pero la diferencia sigue siendo significativa. Se venden el 87,83% que empiezan por 13.         
''')
st.write("")
col1, col2 = st.columns(2)

with col1:
    st.markdown('**Números acados en 13**\n¡Se venden casi todo!')
    st.pyplot(stats.plot('ends_13', 'Termina en 13','x'))
with col2:
    st.markdown('**Números que empiezan en 13**\n')
    st.pyplot(stats.plot('starts_13', 'Empieza en 13', 'x'))
st.markdown('''
#### - Los ceros malditos:
Desconfiamos de los ceros a la izquierda (son unos inútiles). Pero tampoco nos gustan a la derecha, quizás porque resultan en números demasiado redondos.          
''')
st.write("")
col1, col2 = st.columns(2)

with col2:
    st.markdown('**Ventas en función del último dígito**')
    st.pyplot(stats.plot('end_digit', 'Último dígito','x'))
with col1:
    st.markdown('**Ventas en función del primer dígito**\n')
    st.pyplot(stats.plot('start_digit', 'Primer dígito', 'x'))

st.markdown('''
#### - No nos gusta la repetición:
No nos gustan las repeticiones. Pero nos flipa la simetría perfecta. Los números con 5 cifras repetidas, como el 000000 o el 999999 se venden en un porcentaje mayor al 90%.          
''')
st.write("")
col1, col2 = st.columns(2)

with col1:
    st.markdown('**Números repetidos consecutivos**')
    st.pyplot(stats.plot('repeat_consec_max', 'Números repetidos','x'))
with col2:
    st.markdown('** **\n')

st.markdown('''
#### - La belleza del 5 y el 7.
El 5 y el 7 son números bonitos. En general, su presencia mejora las ventas. Pero el efecto es especialmente significativo cuando se enceuntran en el dígito final.      
''')

st.write("")
col1, col2 = st.columns(2)

with col1:
    st.markdown('**Números acabados en 5**')
    st.pyplot(stats.plot('ends_5', 'Termina en 5','x'))
with col2:
    st.markdown('**Números acabados en 7**')
    st.pyplot(stats.plot('ends_7', 'Termina en 7','x'))