execute :
	python3 code/main.py

init :
	mkdir -p model
	mkdir -p result
	mkdir -p data
	
	wget -c https://maups.github.io/tcv3/data_part1.tar.bz2
	tar -jxf data_part1.tar.bz2
	cp -rf ./data_part1/* ./data/
	rm -rf ./data_part1*
