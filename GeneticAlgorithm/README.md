# Algoritmo Genético

Aqui nos foi proposto um trabalho que trabalhasse com **Algoritmos Genéticos**, sendo ele dividido em quatro questões.

Além disso, nesse repositorio encontram-se dois algoritmos, sendo um deles voltado para um problema unidimensional e outro para bidimensional.

---

1) Implemente um Algoritmo Genético para o exemplo de reconhecimento de padrões apresentado em aula. Em vez de reconhecer o número 1 seu algoritmo deve reconhecer o número 0, representado pela bitstring [1 1 1 1 0 1 1 0 1 1 1 1].Teste diferentes taxas de crossover e mutação e compare os resultados. Faça experimentos apenas com crossover e apenas com mutação e compare também os resultados.

2) Implemente um −2((x−0.1)/0.9) 2 Algoritmo Genético para maximizar a função g (x ) = 6 2 (sin(5πx )), já utilizada os exercícios feitos em aula. Utilize uma representação de bitstring. Compare o resultado obtido com os resultados que você obteve com os algoritmos Subida da Colina e Recozimento Simulado aplicados a esta mesma função nos  exercícios feitos em sala de aula. Aproveite para explorar diferentes formas de seleção.

Dica: você também pode aplicar Subida da Colina e Recozimento Simulado em uma bitstring, utilizando uma perturbação  semelhante ao operador de mutação dos algoritmos genéticos, com a vantagem de não ter de se preocupar com o domínio  de x, visto que a própria representação binária dá conta disso.


3) Utilize um Algoritmo Genético para minimizar a seguinte função no intervalo contínuo −5 +5 [ ]: −5 +5 f(x, y) = (1 − x) 2 + 100(y − x 2 ) 2Faça um relatório contendo os dados de seus experimentos, configurações utilizadas, resultados obtidos e suas conclusões. Lembre-se de otimizar os parâmetros dos algoritmos para obter os melhores  resultados e de repetir os experimentos nas mesmas condições diversas vezes para obter uma média e desvio padrão, visto que os algoritmos são estocásticos. Você também pode usar as diferentes versões do algoritmo apresentadas (representação, seleção, etc.).

Além de uma boa aptidão, é desejável que o algoritmo tenha uma convergência rápida, portanto registre também o número de iterações e tempo que os algoritmos demoram para convergir, dados os valores atribuídos aos parâmetros.
Para cada experimento, inclua em seu relatório gráficos que mostrem o valor mínimo e médio da função de aptidão ao longo das iterações.


### Pré-requisito
 - Python3

### Execução
Execute o seguinte comando para rodar em seu computador:
```bash
$ python3 genetic.py
$ python3 genetic_two_dimensions.py
```
