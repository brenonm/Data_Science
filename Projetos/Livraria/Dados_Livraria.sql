--Clientes
CREATE TABLE tb_clientes(
	cpf char(11) PRIMARY KEY
	,nome varchar(50) UNIQUE NOT NULL
	,endereco varchar(30) NOT NULL
	,telefone char(11) UNIQUE NOT NULL
	,email varchar(40) UNIQUE NOT NULL
);

INSERT INTO tb_clientes VALUES ('89874562587','Ana','Rua 1','977854578','ana@gmail.com');
INSERT INTO tb_clientes VALUES ('45784562587','Beatriz','Rua 2','981254578','beatriz@gmail.com');
INSERT INTO tb_clientes VALUES ('35984562587','Juliana','Rua 3','945854578','juliana@hotmail.com');
INSERT INTO tb_clientes VALUES ('12874562587','Carolina','Rua 4','932854578','carolina@gmail.com');
INSERT INTO tb_clientes VALUES ('95674562587','Isabela','Rua 5','958854548','isabela@hotmail.com');
INSERT INTO tb_clientes VALUES ('59878554825','Arthur','Rua 6','988954825','arthur@gmail.com');
INSERT INTO tb_clientes VALUES ('78554873322','Bruno','Rua 4','977587992','bruno@gmail.com');
INSERT INTO tb_clientes VALUES ('22674562587','Joao','Rua 3','955984526','joao@hotmail.com');
INSERT INTO tb_clientes VALUES ('78579854651','Caio','Rua 2','977854298','caio@uol.com');
INSERT INTO tb_clientes VALUES ('79865422545','Ian','Rua 1','989854588','ian@gmail.com');
INSERT INTO tb_clientes VALUES ('89988554222','Amanda','Rua 5','955684425','amanda@gmail.com');
INSERT INTO tb_clientes VALUES ('44877478855','Bianca','Rua 7','977452219','bianca@gmail.com');
INSERT INTO tb_clientes VALUES ('45512445722','Joana','Rua 2','977885933','joana@uol.com');
INSERT INTO tb_clientes VALUES ('55475622365','Cinthia','Rua 3','932552366','cinthia@hotmail.com');
INSERT INTO tb_clientes VALUES ('44487556998','Isis','Rua 8','978885426','isis@gmail.com');
INSERT INTO tb_clientes VALUES ('89866522178','Mariana','Rua 8','958854578','mariana@hotmail.com');
INSERT INTO tb_clientes VALUES ('45665488922','Andre','Rua 1','977889854','andre@gmail.com');
INSERT INTO tb_clientes VALUES ('78898754623','Gabriela','Rua 3','945216653','gabriela@gmail.com');
INSERT INTO tb_clientes VALUES ('44587854211','Rafael','Rua 2','977986521','rafael@gmail.com');
INSERT INTO tb_clientes VALUES ('66844522432','Leticia','Rua 5','988785425','leticia@gmail.com');

SELECT * FROM tb_clientes

--Vendas
CREATE TABLE tb_vendas(
	id smallint GENERATED ALWAYS AS IDENTITY
	,cpf char(11) NOT NULL
	,titulo varchar(50) NOT NULL
	,quantidade smallint NOT NULL
	,FOREIGN KEY(cpf) REFERENCES tb_clientes(cpf)
	
);

INSERT INTO tb_vendas(cpf,titulo,quantidade)VALUES('89874562587','A Primeira Aventura',5);
INSERT INTO tb_vendas(cpf,titulo,quantidade)VALUES('45784562587','A Segunda Aventura',2);
INSERT INTO tb_vendas(cpf,titulo,quantidade)VALUES('35984562587','A Terceira Aventura',1);
INSERT INTO tb_vendas(cpf,titulo,quantidade)VALUES('12874562587','A Quarta Aventura',6);
INSERT INTO tb_vendas(cpf,titulo,quantidade)VALUES('95674562587','A Quinta Aventura',1);
INSERT INTO tb_vendas(cpf,titulo,quantidade)VALUES('59878554825','O Primeiro Drama',2);
INSERT INTO tb_vendas(cpf,titulo,quantidade)VALUES('78554873322','O Primeiro Drama',1);
INSERT INTO tb_vendas(cpf,titulo,quantidade)VALUES('22674562587','A Primeira Aventura',1);
INSERT INTO tb_vendas(cpf,titulo,quantidade)VALUES('78579854651','A Primeira Aventura',1);
INSERT INTO tb_vendas(cpf,titulo,quantidade)VALUES('79865422545','A Segunda Aventura',2);
INSERT INTO tb_vendas(cpf,titulo,quantidade)VALUES('89988554222','A Segunda Aventura',1);
INSERT INTO tb_vendas(cpf,titulo,quantidade)VALUES('44877478855','A Terceira Aventura',1);
INSERT INTO tb_vendas(cpf,titulo,quantidade)VALUES('45512445722','A Quarta Aventura',1);
INSERT INTO tb_vendas(cpf,titulo,quantidade)VALUES('55475622365','O Primeiro Drama',1);
INSERT INTO tb_vendas(cpf,titulo,quantidade)VALUES('44487556998','O Segundo Drama',2);
INSERT INTO tb_vendas(cpf,titulo,quantidade)VALUES('89866522178','O Primeiro Suspense',2);
INSERT INTO tb_vendas(cpf,titulo,quantidade)VALUES('45665488922','O Primeiro Suspense',3);
INSERT INTO tb_vendas(cpf,titulo,quantidade)VALUES('78898754623','O Segundo Suspense',4);
INSERT INTO tb_vendas(cpf,titulo,quantidade)VALUES('44587854211','O Terceiro Suspense',4);
INSERT INTO tb_vendas(cpf,titulo,quantidade)VALUES('66844522432','O Primeiro Romance',1);
INSERT INTO tb_vendas(cpf,titulo,quantidade)VALUES('89874562587','A Segunda Aventura',1);
INSERT INTO tb_vendas(cpf,titulo,quantidade)VALUES('45784562587','A Terceira Aventura',1);
INSERT INTO tb_vendas(cpf,titulo,quantidade)VALUES('35984562587','A Quarta Aventura',1);
INSERT INTO tb_vendas(cpf,titulo,quantidade)VALUES('95674562587','O Primeiro Drama',1);
INSERT INTO tb_vendas(cpf,titulo,quantidade)VALUES('59878554825','O Primeiro Suspense',2);
INSERT INTO tb_vendas(cpf,titulo,quantidade)VALUES('78554873322','A Primeira Aventura',1);
INSERT INTO tb_vendas(cpf,titulo,quantidade)VALUES('22674562587','A Segunda Aventura',1);
INSERT INTO tb_vendas(cpf,titulo,quantidade)VALUES('78579854651','A Segunda Aventura',2);
INSERT INTO tb_vendas(cpf,titulo,quantidade)VALUES('79865422545','A Terceira Aventura',3);


SELECT * FROM tb_vendas;

--Livros
CREATE TABLE tb_livros(
	isbn char(10) PRIMARY KEY
	,titulo varchar(50) UNIQUE NOT NULL
	,genero char (30) NOT NULL
	,ano int NOT NULL
	,preco real NOT NULL
);

INSERT INTO tb_livros VALUES('7896541278','A Primeira Aventura','Comédia',1989,45.50);
INSERT INTO tb_livros VALUES('9896541278','A Segunda Aventura','Suspense',1995,35.90);
INSERT INTO tb_livros VALUES('4796541278','A Terceira Aventura','Comédia',2000,50);
INSERT INTO tb_livros VALUES('6396541278','A Quarta Aventura','Ficção',2002,30);
INSERT INTO tb_livros VALUES('3496541278','A Quinta Aventura','Ficção',2004,52);
INSERT INTO tb_livros VALUES('3096541278','O Primeiro Drama','Romance',1989,60);
INSERT INTO tb_livros VALUES('5568944565','O Segundo Drama','Romance',1995,65);
INSERT INTO tb_livros VALUES('3798989898','O Primeiro Suspense','Ficção',1950,55);
INSERT INTO tb_livros VALUES('3453899989','O Segundo Suspense','Ficção',1976,60.40);
INSERT INTO tb_livros VALUES('3453555555','O Terceiro Suspense','Romance',1995,70);
INSERT INTO tb_livros VALUES('4658444596','O Primeiro Romance','Comédia',2000,80.50);

SELECT * FROM tb_livros;


--Filtrar clientes por email
SELECT * FROM tb_clientes WHERE email LIKE '%@gmail.com';
SELECT * FROM tb_clientes WHERE email LIKE '%@hotmail.com';
SELECT * FROM tb_clientes WHERE email LIKE '%@uol.com';

--Contar frequência de email
SELECT
	SPLIT_PART(email,'@',2) AS tipo_email
	,COUNT (email)
FROM tb_clientes
GROUP BY 1;

--Filtrar produtos únicos entre todas as vendas
SELECT DISTINCT titulo FROM tb_vendas;

--Ordenar livros por ano de publicação
SELECT titulo,genero,ano,preco FROM tb_livros ORDER BY ano;

--Contar frequência gêneros literários
SELECT
	genero
	,COUNT(genero) AS unidades
FROM tb_livros
GROUP BY 1
ORDER BY unidades DESC;



--Vendas por livro
SELECT
	A.cpf
	,A.quantidade
	,B.titulo
	,B.ano
	,B.preco
	,(A.quantidade * B.preco) AS receita
FROM tb_vendas AS A
LEFT JOIN tb_livros AS B
ON A.titulo = B.titulo
ORDER BY receita DESC;

--Filtrar vendas por livro específico com nome cliente
SELECT
	C.nome
	,B.titulo
	,A.quantidade
	,B.preco
	,(A.quantidade * B.preco) as receita
FROM tb_vendas AS A
LEFT JOIN tb_livros AS B
ON A.titulo = B.titulo
INNER JOIN tb_clientes AS C
ON A.cpf = C.cpf
WHERE A.titulo = 'A Primeira Aventura';

--Top 3 gêneros literários mais vendidos
SELECT
	B.genero
	,SUM(A.quantidade) AS unidades_vendidas
FROM tb_vendas AS A
LEFT JOIN tb_livros AS B
ON A.titulo = B.titulo
GROUP BY B.genero
ORDER BY unidades_vendidas DESC
LIMIT 3;

--Receita por cliente
SELECT
	C.nome
	,SUM(A.quantidade * B.preco) AS receita_cliente
FROM tb_vendas AS A
INNER JOIN tb_livros AS B
ON A.titulo = B.titulo
INNER JOIN tb_clientes AS C
ON A.cpf = C.cpf
GROUP BY C.nome
ORDER BY receita_cliente DESC;

--Vendas por cliente
SELECT
	B.nome
	,A.titulo
	,A.quantidade
FROM tb_vendas AS A
INNER JOIN tb_clientes AS B
ON A.cpf = B.cpf
ORDER BY nome;

--Verificar quantas vezes clientes compraram
SELECT
	B.nome
	,COUNT(A.cpf)
FROM tb_vendas AS A
INNER JOIN tb_clientes AS B
ON A.cpf = B.cpf
GROUP BY B.cpf
HAVING COUNT(A.cpf) >= 1;
