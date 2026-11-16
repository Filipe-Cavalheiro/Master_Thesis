# Notas do orador — defesa de mestrado (objetivo ≈ 19 min, limite 20 min)

Cada bloco `#### [Cena.k]` é um sub-slide (uma tecla). O `slides.py` lê estes blocos ao renderizar
e anexa-os como notas do orador do manim-slides (tecla `S` na exportação HTML abre a vista do
orador). Manter o formato dos títulos ao editar o texto.

## Orçamento de tempo

| # | Cena | Sub-slides | Tempo | Acumulado |
|---|---|---|---|---|
| 1 | TitleCard | 1 | 0:25 | 0:25 |
| 2 | Motivation | 3 | 1:10 | 1:35 |
| 3 | Problem | 3 | 1:15 | 2:50 |
| 4 | Roadmap | 1 | 0:35 | 3:25 |
| 5 | StateOfTheArt | 1 | 0:45 | 4:10 |
| 6 | Background | 4 | 1:30 | 5:40 |
| 7 | CentralizedDistribution | 2 | 1:05 | 6:45 |
| 8 | CentralizedControl | 3 | 1:50 | 8:35 |
| 9 | CentralizedResults | 3 | 2:00 | 10:35 |
| 10 | Decentralized | 4 | 1:50 | 12:25 |
| 11 | GeneralizedMesh | 5 | 1:30 | 13:55 |
| 12 | GeneralizedMPC | 3 | 1:30 | 15:25 |
| 13 | Results | 12 | 3:40 | 19:05 |
| 14 | Conclusions | 1 | 0:55 | 20:00 |
| 15 | FutureWork | 1 | 0:30 | 20:30 |
| 16 | ThankYou | 1 | 0:10 | 20:40 |
| — | Extra (só nas questões) | 11 | — | — |

≈ 20:40 de fala: **acima do limite de 20 min**. Os dois vídeos dos testes reais passam em tempo
real e completos (46 s e 72 s). Para caber, avançar os vídeos com a tecla de seguinte assim que a
ideia estiver dita (a cena seguinte começa logo), ou encurtar a análise quantitativa dos
Resultados e a ponderação pela velocidade (uma frase cada). Se estiver atrasado no slide
**Decentralized** (objetivo: 12:25), encurtar também a tabela invólucro convexo vs semiplano.

Pontos de controlo: Roadmap concluído por ~3:30, Centralizado por ~10:35, Resultados por ~19:05.

---

## 1. TitleCard (0:25)

#### [TitleCard.1] Título
Bom dia. O meu nome é Filipe Cavalheiro e vou apresentar a minha dissertação de mestrado,
"Cerco Cooperativo de Drones Não Cooperativos", orientada pelo Professor Bruno Guerreiro.
Em resumo: como é que um enxame de drones pode circundar um drone intruso em 3D, em segurança e
sem o destruir.

## 2. Motivação (1:10)

#### [Motivation.1] Notícias
Os drones já não são apenas uma ferramenta de investigação ou militar — qualquer pessoa pode
comprar um. E quando um drone é avistado perto de um aeroporto, as operações param. Estes são dois
exemplos recentes: Bruxelas, e aeroportos na Bélgica e na Suécia encerrados por causa de drones.

#### [Motivation.2] Números da DFS
Isto não são casos isolados. Só na Alemanha, o controlo de tráfego aéreo registou 161
perturbações causadas por drones em 2024 e 225 em 2025. Cada uma significa voos atrasados,
desviados ou cancelados e, junto a heliportos hospitalares, pode atrasar o transporte de
emergência. Muitas destas incursões são não intencionais: o operador simplesmente não conhece as
regras ou a zona restrita.

#### [Motivation.3] Pergunta
A pergunta desta tese é então: como circundar um drone que entra em espaço aéreo restrito de
forma segura e visível, e sem meios destrutivos, para que o operador perceba que tem de sair?

## 3. Problema (1:15)

#### [Problem.1] Três noções de circundamento
"Circundamento" significa coisas diferentes na literatura. Alguns trabalhos exigem que o alvo
esteja dentro do invólucro convexo dos perseguidores; outros exigem que os conjuntos alcançáveis
se sobreponham. Aqui queremos a terceira: uma formação visível em torno, ou à frente, do alvo. É
um aviso e uma barreira, não uma captura.

#### [Problem.2] Objetivo 3D
O alvo, a vermelho, é não cooperativo — só temos uma medição com ruído da sua posição. Cada
perseguidor i deve convergir para um ponto desejado p_i-estrela em torno dele. Os requisitos:
totalmente 3D, alvo estático ou móvel, agentes o mais uniformemente espaçados possível — porque
não sabemos para onde o alvo vai, e o espaçamento uniforme cobre todas as direções por igual —,
movimento com prevenção de colisões e, idealmente, um sistema descentralizado que sobreviva à
perda de agentes.

#### [Problem.3] Âmbito
O que fica fora do âmbito: operadores maliciosos que ignorem a formação, captura ou interceção
física, e o procedimento legal que se segue ao circundamento.

## 4. Roadmap (0:35)

#### [Roadmap.1] Três etapas
O trabalho está construído em três etapas, cada uma assente na anterior. Cada etapa responde a
três perguntas: para onde deve ir cada agente, como lá chega e como o mantemos seguro.
Centralizado: rede de Fibonacci, controlador geométrico de Lyapunov e uma salvaguarda APF.
Descentralizado: a rede é substituída por um algoritmo de Lloyd/Voronoi local na esfera.
Generalizado: qualquer malha de superfície e um único MPC com polinómios de Bernstein para o
planeamento de trajetórias. Em todas, a estimativa do alvo vem de medições com ruído, de uma
média móvel e de consenso.

## 5. Estado da arte (0:45)

#### [StateOfTheArt.1] Abordagens
Três famílias dominam o controlo de formações circulares. Perseguição cíclica: cada agente
persegue o seguinte; muito pouca comunicação, mas sem termos adicionais os agentes colapsam num
ponto. Líder–seguidor: intuitivo, mas o líder é um ponto único de falha. Campos de potencial
artificial: elegantes e baseados em Lyapunov, mas com mínimos locais e, para formações, o número
de agentes tem de ser conhecido. A lacuna principal: quase tudo isto é planar, e os trabalhos 3D
atribuem os agentes a setores predefinidos.

## 6. Fundamentos (1:30)

#### [Background.1] Algoritmo de Lloyd
A ferramenta que preciso que tenham presente é o algoritmo de Lloyd. Dados os sítios, divide-se
o domínio de modo que cada ponto pertença ao sítio mais próximo — as células de Voronoi. Depois
move-se cada sítio para o centroide da sua célula, e repete-se. Converge para uma tesselação
centroidal de Voronoi: uma configuração uniformemente distribuída. As etapas descentralizada e
generalizada assentam ambas nisto.

#### [Background.2] Vizinhos de Delaunay
O dual do diagrama de Voronoi é a triangulação de Delaunay: dois agentes são vizinhos se
partilharem uma aresta de Voronoi. Uso-a como grafo de comunicação — ao contrário dos k vizinhos
mais próximos ou de um raio fixo, adapta-se ao movimento dos agentes e mantém-se conexo.

#### [Background.3] Consenso
Neste grafo, cada agente faz a média da sua estimativa do alvo com as dos vizinhos. Partindo de
medições com ruído, todas as estimativas convergem para um valor comum — sem nó central.

#### [Background.4] Métodos de segurança
Para as colisões há três famílias. Os APF atuam na referência, sem garantia de invariância —
usados nos capítulos 4 e 5. Os CBF-QP dão invariância positiva de um conjunto seguro, mas atuam
na entrada instantânea — não usados. O MPC da tese restringe as posições previstas ao longo de um
horizonte, juntamente com limites de velocidade e aceleração. O CBF-MPC, que combina ambos, é
trabalho futuro.

## 7. Centralizado — distribuição (1:05)

#### [CentralizedDistribution.1] Rede de Fibonacci
No caso centralizado, um nó conhece todos os agentes, pelo que podemos calcular a formação
diretamente. Uso a rede de Fibonacci: cada ponto tem a sua latitude, e pontos consecutivos são
rodados pelo ângulo de ouro, cerca de 137,5 graus. Como a razão de ouro é o número "mais
irracional", os pontos nunca se alinham e o espaçamento mantém-se uniforme para qualquer tamanho.

#### [CentralizedDistribution.2] Comparação
Comparei-a com uma grelha latitude–longitude e com a amostragem por disco de Poisson, com cerca
de 1000 pontos. A lat–lon é a mais regular, mas só produz formações com o tamanho de uma grelha.
O disco de Poisson é irregular, com coeficiente de variação 0,25. Fibonacci: CV 0,0105, a menor
energia de Riesz e cerca de 700 vezes mais rápido que o disco de Poisson — 0,1 milissegundos. A
única restrição é um número ímpar de pontos.

## 8. Centralizado — controlo (1:50)

#### [CentralizedControl.1] Decomposição
Para chegar ao ponto, divido a posição relativa ao alvo numa distância r_i e numa direção q_i na
esfera unitária. O objetivo passa a ser dois subobjetivos desacoplados: r_i para r-estrela e q_i
para q_i-estrela.

#### [CentralizedControl.2] Função de erro
Para a direção, o erro clássico é e_r = S(q-estrela) q, o produto externo de q-estrela e q. A sua
norma é sen(teta): atinge o máximo aos 90 graus e volta a zero junto ao antípoda — o controlador
quase não veria erro quando está o mais errado possível. O erro proposto por Ramp et al., que uso,
divide-o por raiz de 2(1 + q·q-estrela) e tem norma sen(teta/2): é monótono e nunca indica
falsamente convergência.

#### [CentralizedControl.3] Leis de controlo e teorema
O controlador geométrico roda q ao longo da esfera com ganho k_geo. O controlador radial reduz a
distância, mas o seu ganho é multiplicado por exp(-|e_q|²): os agentes primeiro alinham-se e
depois aproximam-se — vê-se na animação, incluindo um agente que começa quase no antípoda. Com
V = Psi como função de Lyapunov, a derivada de V é menos k_geo |e_q|² e, com LaSalle em conjuntos
de subnível, o equilíbrio é assintoticamente estável para todas as direções iniciais exceto o
antípoda — quase global. O erro radial decai exponencialmente com taxa de pelo menos k_r sobre e.

## 9. Centralizado — segurança e resultados (2:00)

#### [CentralizedResults.1] Salvaguarda APF
Este controlador, por si só, não evita colisões enquanto os agentes atravessam a esfera. Por
isso, uma salvaguarda APF modifica apenas a referência: se dois agentes estiverem mais próximos
do que d_min, metade do espaçamento da rede, é adicionada uma repulsão tangencial e o resultado é
reprojetado na esfera. À esquerda sem, à direita com: os agentes desviam-se um do outro. Os
pontos da rede são atribuídos com o algoritmo de Munkres, minimizando a distância geodésica total.

#### [CentralizedResults.2] 31 agentes, alvo móvel
Resultados com 31 agentes e um alvo móvel com 5 % de ruído de medição. A formação constitui-se e
segue o alvo. Com o alvo móvel há uma oscilação limitada em torno de r-estrela (2 m) e um erro de
seguimento de cerca de um metro, porque o controlador não usa a velocidade do alvo. A distância
entre agentes mantém-se sempre acima do mínimo.

#### [CentralizedResults.3] Teste real
Este método foi também testado em voo com Crazyflies e OptiTrack. Aqui, três drones executam o
método centralizado, rede de Fibonacci mais controlador geométrico, em torno de um alvo virtual
— a magenta — que se move numa circunferência. A verde está o drone real e a laranja a simulação
do mesmo voo desenhada sobre a imagem da câmara; diferem cerca de 10 centímetros RMS. O vídeo
está em tempo real, da descolagem à aterragem (46 s); a tecla seguinte avança quando quiser.

## 10. Descentralizado (1:50)

#### [Decentralized.1] Lloyd local na esfera
Descentralizar quebra duas coisas: a rede de Fibonacci precisa do número total de agentes, e o
Munkres precisa da posição de todos. Assim, cada agente constrói a sua célula de Voronoi a partir
dos vizinhos e move o seu objetivo para o centroide — o algoritmo de Lloyd na esfera, com
distância geodésica. Aqui, 12 agentes espalham-se e a variação do espaçamento diminui.

#### [Decentralized.2] Perda de agentes
Agora perdem-se quatro agentes. Ninguém replaneia globalmente: os oito restantes continuam
simplesmente a executar o Lloyd e redistribuem-se por toda a esfera. O tamanho da formação pode
mudar sem qualquer reprojeto.

#### [Decentralized.3] Invólucro convexo vs semiplano
Implementei as células de Voronoi de duas formas. A versão com invólucro convexo é rápida —
0,03 s por execução — e convergiu sempre, mas numa esfera fechada pode escolher o arco errado e
criar células degeneradas. A versão de semiplano constrói cada célula a partir de restrições de
bissetriz: mais uniforme, CV 0,073 contra 0,120, sem casos degenerados, mas mais lenta — 1,37 s —
e convergiu em 93 de 100 execuções.

#### [Decentralized.4] Resultados
12 agentes, alvo móvel, raio desejado de 2 m, 4 perdidos aos 10 segundos, semiplano. Com o alvo
em movimento, os agentes aproximam-se da esfera mais devagar do que com o alvo estático: aos 30 s a
distância média ainda é cerca de 4,3 m, porque o controlador não usa a velocidade do alvo. Mas a
distribuição dos pontos desejados mantém-se uniforme — o CV fica abaixo de 0,1 e o espaçamento
passa de 2,0 para 2,2 m depois da perda — e a distância mínima entre agentes fica em cerca de
1,7 m, sem quase-colisões. Não é preciso reprojetar nada quando o tamanho da formação muda.

## 11. Generalizado — malhas e controlador de malha (1:30)

#### [GeneralizedMesh.1] Qualquer malha
Uma esfera é apenas uma forma. Tratando a malha como um grafo pesado, o Dijkstra dá geodésicas
por caminho mais curto, e o Lloyd funciona em qualquer malha bem definida — aqui um busto e um
sapo do conjunto de dados Thingi10K, com 8 sítios cada.

#### [GeneralizedMesh.2] Controlador de malha
Isto permite o comportamento principal. Longe da fronteira, os agentes formam um disco plano:
uma barreira, um aviso à frente da zona restrita. A malha é uma interpolação entre três formas
pré-calculadas — disco, cone e esfera — comandada pela distância com sinal do alvo à fronteira.

#### [GeneralizedMesh.3] Disco para cone
À medida que o alvo se aproxima e atravessa a fronteira, o disco deforma-se num cone em torno
dele; o cone fica completo quando o alvo está meio raio para lá da fronteira.

#### [GeneralizedMesh.4] Cone para esfera
Mais para dentro, o cone fecha-se numa esfera — um envolvimento — completa quando o alvo está um
raio para lá da fronteira; a partir daí a esfera segue o alvo.

#### [GeneralizedMesh.5] O alvo sai
E se o alvo sair, o processo inverte-se e a formação regressa à barreira.

## 12. Generalizado — MPC e velocidade (1:30)

#### [GeneralizedMPC.1] MPC de Bernstein
Os pontos já não estão numa esfera, pelo que o controlador geométrico é substituído por um MPC.
Minimiza a distância ao objetivo de Lloyd mais um termo de aceleração, sujeito à consistência do
estado inicial, a limites de velocidade e aceleração e a um envelope elipsoidal de colisão com os
vizinhos, cujos horizontes previstos são partilhados pela rede. A trajetória é um polinómio de
Bernstein de grau 5, pelo que as derivadas são lineares nos pontos de controlo. Só o primeiro
passo é aplicado, e depois resolve-se de novo. Uma nota honesta: o termo de colisão é uma
penalização elevada, não uma restrição rígida, pelo que não há garantia formal de segurança.

#### [GeneralizedMPC.2] Sem velocidade: CVT uniforme
Por fim, a velocidade do alvo. Sem ela, os 12 pontos de Lloyd numa malha esférica distribuem-se
uniformemente: as cores são as células de Voronoi calculadas com Dijkstra. O alinhamento
direcional médio é praticamente zero e metade dos agentes está à frente do alvo.

#### [GeneralizedMPC.3] Com velocidade: CVT ponderada
Quando a velocidade é conhecida, cada vértice recebe um peso que cresce com o ângulo à direção
atrás do alvo, e as arestas da malha ficam mais longas à frente. As células da frente encolhem e
o algoritmo de Lloyd volta a convergir em poucas iterações, com mais agentes à frente. Com sigma
1,2 e 12 agentes, o alinhamento sobe para cerca de 0,32 e 8 de 12 agentes ficam no hemisfério da
frente, como na tese.

## 13. Resultados (3:40)

#### [Results.1] Repetição: disco na fronteira
Estes são os dados reais da simulação, ao dobro da velocidade: 12 agentes, o alvo voa numa
circunferência, entrando e saindo da região restrita (x > 0). Cada tecla avança até à fase
seguinte. Ao 1 s, os agentes formam um disco na fronteira.

#### [Results.2] Cone
Aos 2,5 s, com o alvo a atravessar a fronteira, a malha é um cone.

#### [Results.3] Esfera fechada
Aos 4 s, o alvo está dentro e a malha é uma esfera fechada em torno dele.

#### [Results.4] Perda de agentes
Aos 10 s, quatro agentes ficam sem energia e caem — os outros oito redistribuem-se na esfera.

#### [Results.5] Regresso à barreira
Aos 13 s, o alvo sai e a formação regressa à barreira plana.

#### [Results.6] Fim da simulação
E espera aí até ao fim da simulação.

#### [Results.7] Análise quantitativa
Após o transitório inicial, a distância mínima entre agentes mantém-se entre 3 e 4 metros, bem
acima do limiar de 1 m. As velocidades respeitam sempre o limite de 3 m/s. A formação recupera
após a perda. Dois pontos fracos: um agente passa muito perto do alvo entre os 2 e os 5 s, porque
o MPC não tem restrição agente–alvo; e o erro ao objetivo chega a cerca de 4 m quando os agentes
estão limitados em velocidade, com um perfil em escada porque a atualização de Voronoi corre a
1 Hz.

#### [Results.8] Teste real
E aqui quatro drones executam o método generalizado: a CVT geodésica na malha — os pontos azuis —
dá os objetivos em torno do alvo virtual, a magenta, e o MPC de Bernstein planeia as trajetórias.
Descolagem, aproximação, e depois a formação segue o alvo. As trajetórias reais e simuladas
coincidem a poucos centímetros. No regresso, o CF02 real cai; a simulação completa o seu voo e o
anel vermelho marca a última posição registada. O vídeo está em tempo real (72 s); a tecla
seguinte avança quando quiser.

#### [Results.9] Aeroporto de Lisboa: barreira na parede
Por fim, a mesma arquitetura sobre a zona proibida real do aeroporto de Lisboa, a partir dos
dados oficiais da ANAC; é uma simulação com o modelo do X500, a 3x. Os doze perseguidores mantêm
uma barreira na parede, 30 metros no exterior da zona, e deslizam com o alvo; aos 13,9 s o alvo
atravessa a parede e fecham a esfera.

#### [Results.10] Perda de perseguidores
Aos 25 segundos, quatro perseguidores ficam sem energia e os outros oito redistribuem-se.

#### [Results.11] O alvo entra na zona proibida
Aos 34,1 s o alvo entra na zona proibida e a formação achata-se numa barreira na fronteira.

#### [Results.12] Barreira na fronteira
E fica a um metro dela até ao fim: os perseguidores nunca entram na zona.

## 14. Conclusões (0:55)

#### [Conclusions.1] Conclusões
Para concluir: um enxame consegue circundar um alvo não cooperativo em 3D, em simulação e em
voos reais com Crazyflies.
Centralizado: a rede de Fibonacci com o controlador de Lyapunov é a opção mais simples, escala
até 31 agentes e tem uma prova de convergência quase global. Descentralizado: consenso mais Lloyd
local funciona sem nó central e sobrevive à perda de agentes. Generalizado: qualquer malha,
barreira e envolvimento, um único MPC com limites dinâmicos. As limitações são claras: voos reais
apenas em interior e com alvo virtual, comunicação instantânea, posição real do alvo mais ruído, e colisões penalizadas em
vez de formalmente excluídas.

## 15. Trabalho futuro (0:30)

#### [FutureWork.1] Trabalho futuro
Próximos passos: passar para hardware real com PX4 e ROS 2, comunicação e perceção realistas; e,
do lado do método, usar a velocidade do alvo no MPC, formações baseadas em alcançabilidade e um
CBF-MPC para garantias formais de segurança.

## 16. Obrigado (0:10)

#### [ThankYou.1] Obrigado
Obrigado pela vossa atenção. Terei todo o gosto em responder às vossas questões.

#### [ThankYou.2] Questões (ciclo)
(Animação em ciclo enquanto decorrem as questões.)

---

## Slides extra (apenas durante as questões)

#### [BackupLyapunov.1] Esboço da prova
Argumento completo de estabilidade: Psi é contínua e limitada em S², a derivada de V é ≤ 0, os
conjuntos de subnível D_c com c < 2 são compactos, positivamente invariantes e excluem o
antípoda; LaSalle em cada D_c; os D_c esgotam S² menos o antípoda. Subsistema radial: cascata,
com taxa limitada inferiormente por k_r/e porque |e_q|² = sen²(θ/2) < 1.

#### [BackupControllers.1] Comparação de controladores
Porquê Lyapunov em vez de fmincon e APF (3 agentes, r* = 2 m, 30 s): o fmincon nunca chega a
0,1 m em 30 s; o APF converge em 4,46 s, mas precisa de conhecer N à partida e não tem garantia
de estabilidade como método de distribuição; o Lyapunov demora 17,6 s (primeiro alinhar, depois
aproximar), mas tem erro de seguimento quase nulo e o menor tempo de computação (7,8 s).

#### [BackupDirect.1] Métodos de cálculo direto
Comparação dos três métodos diretos com cerca de 1000 pontos: a grelha latitude–longitude é a mais
regular mas só existe para tamanhos de grelha; o disco de Poisson é irregular (CV 0,25); a rede
de Fibonacci tem espaçamento quase uniforme (CV 0,011) e a menor energia de Riesz.

#### [BackupDirect.2] Tempo de execução
Em 100 ensaios, a rede de Fibonacci demora cerca de 0,1 ms, cerca de 700 vezes menos que o disco
de Poisson, e com muito menor dispersão.

#### [BackupIterative.1] Métodos de cálculo iterativo
Os três métodos iterativos com 30 pontos e 100 execuções: o Thomson (energia mínima) nunca
convergiu no critério e tem o pior espaçamento; o Lloyd com invólucro convexo converge sempre em
cerca de 20 iterações; o Lloyd com semiplanos é o mais uniforme (CV 0,073) mas converge em 93 de
100 execuções.

#### [BackupIterative.2] Tempo de execução
O semiplano é cerca de 40 vezes mais lento que o invólucro convexo; por isso os dois foram
testados no sistema completo.

#### [BackupDecentralized.1] Falha do invólucro convexo
Na esfera fechada, o invólucro convexo pode escolher o arco de círculo máximo mais curto quando a
fronteira de Voronoi verdadeira é o mais longo; o Lloyd move então os sítios para uma partição
errada e vários sítios podem colapsar juntos. A construção por semiplanos evita isto.

#### [BackupArchitectures.1] Arquitetura centralizada
Um nó central calcula a rede de Fibonacci, atribui os pontos com o Munkres e envia a cada agente
o seu ponto; cada agente segue-o com o controlador geométrico e radial e a salvaguarda APF.

#### [BackupArchitectures.2] Arquitetura descentralizada
Sem nó central: cada agente estima o alvo por consenso na rede de Delaunay, calcula a sua célula
de Voronoi a partir dos vizinhos e segue o centroide com o mesmo controlador.

#### [BackupArchitectures.3] Arquitetura generalizada e parâmetros
Medição com ruído → EMA → consenso na rede espacial de Delaunay → controlador de malha → Lloyd
(Dijkstra) → MPC → controlador de Mellinger → modelo do quadrirrotor. Parâmetros conforme
listados (tabela da tese, Cap. 6).
