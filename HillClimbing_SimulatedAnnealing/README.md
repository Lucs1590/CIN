# Search methods
Aqui nos foi proposto a realização de um trabalho que trabalhasse com os algoritimos **simulated annealing** e **hill climbing**, além de ser possivel realizar variações dos mesmos.

Os conceitos aprendidos aqui são fruto das aulas do [Prof. Dr. Fabricio Breve](https://github.com/fbreve).

---

## Questões

1. Implemente os vários tipos de algoritmos de Subida da Colina e Recozimento Simulado para resolver o problema proposto no Exemplo de Aplicação. Use um esquema de representação para as soluções candidatas (variável x). Compare o desempenho dos algoritmos e tire suas conclusões.
2. Para o Subida da Colina simples, utilize diferentes configurações iniciais como tentativas de encontrar o ótimo global. O algoritmo teve sucesso?
3. Discuta a sensibilidade de todos os algoritmos com relação aos seus parâmetros de entrada.

---

## Respostas

1. O algoritmo se encontra [aqui](https://raw.githubusercontent.com/Lucs1590/CIN/master/HillClimbing_SimulatedAnnealing/search_methods.py?token=AJGLCQYLDMGKYG3XLFFCD3K6VIVYC)!
2. Como o algoritmo executa com uma semente aleatória ficamos muito dependente do valor que será o nosso primeiro numero. Em certos casos o algoritmo conseguiu atingir 0.933 (com uma semente de 1587603556.2999938), todavia o importante é que esse algoritmo tem como foco atingir o maximo local e se isso não é possivel, ele estagna, formando uma especie de constante.
Além da semente randomica, podemos alterar a quantidade de inteções, todavida quando o algoritmo não consegue evoluir, comumente é gera-se uma constante (denominada de shoulder, segundo a figura abaixo) e para que o custo custo computacional seja reduzido, considera-se esse valor como melhor local, após uma quantidade x de interações.

![hc_esquema](https://static.javatpoint.com/tutorial/ai/images/hill-climbing-algorithm-in-ai.png)

3. Quanto a sensibilidade dos algoritimos, vale dizer que o ponto inicial tem extrema importancia, depois os ajustes que são feitos nessa algoritmo proporcionam um bom desempenho. Por exemplo, a função que avaliará o custo dos pontos. Digo que o coração do algoritmo se encontra ali, pois a partir dela que julgaremos o caminho que o algoritmo irá tomar, se será para maximizar ou minimizar e afins.
Outro ponto de extrema importancia é a perturbação que será feita nos pontos. Como dito em aula, uma perturbação em demasia, poderia perder um otimo global, porém provocações pequenas necessitam de uma quantidade maior de interações para que seja possivel achar um ótimo melhor. O importante é saber que ambas situações devem ser consideradas.
Por fim, embora em meus testes foi observado uma diferenção não muito signitificativa é visivel que com o **simulated annealing** foi possível encontrar menos interações e um resultado bem alto, visto que o mesmo possui uma probabilidade maior de coletar valores maiores com a utilização da probabilidade euleriana.

---

## Exemplos de execuções

---

### Exemplo 1
Hill Climbing

![hc1](https://i.imgur.com/bVqmuNd.png)

Simulated Annealing

![sm1](https://i.imgur.com/NM2xoC3.png)

| Método              | Resultado x        | Custo              |
|---------------------|--------------------|--------------------|
| Hill Climbing       | 0.13               | 0.4995930048308231 |
| Simulated Annealing | 0.2992266636607204 | 0.9339117930977985 |

---

### Exemplo 2
Hill Climbing

![hc2](https://i.imgur.com/T2EjpSG.png)

Simulated Annealing

![sm2](https://i.imgur.com/7VZ5hsv.png)

| Método              | Resultado x         | Custo              |
|---------------------|---------------------|--------------------|
| Hill Climbing       | 0.11                | 0.9282078435903955 |
| Simulated Annealing | 0.29970235273480167 | 0.9339606751076133 |

---

### Exemplo 3
Hill Climbing

![hc3](https://i.imgur.com/jY8AWYu.png)

Simulated Annealing

![sm3](https://i.imgur.com/itEgtSC.png)

| Método              | Resultado x         | Custo              |
|---------------------|---------------------|--------------------|
| Hill Climbing       | 0.10019150787713275 | 0.9999727897822276 |
| Simulated Annealing | 0.10020041028989954 | 0.999970201234957  |