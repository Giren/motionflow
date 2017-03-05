#include <stdio.h>
#include <stdlib.h>

unsigned char * readPGM(const char * filename, int width, int height)
{
	FILE *fd = fopen(filename, "rb");
	int LF_c = 0;
	char buffer;
	printf("loading %s...\n", filename);
	do {
		buffer = (char)fgetc(fd);
		printf("%c",buffer);
		if(buffer == 0xA) {
			LF_c++;
		}
	} while(buffer != EOF && LF_c < 4);
    //Header abschneiden -> 4xLF
    
	unsigned char * raw_image = (unsigned char *)malloc(width*height);
	if(raw_image) {
		fread(raw_image,1,width*height,fd);
		printf("loaded %s\n", filename);
	} else {
		printf("error loading %s\n", filename);
		return NULL;
	}
	return raw_image;
}

void * writePGM(const char * filename, unsigned char * buffer, int width, int height) {
	FILE *fd = fopen(filename, "w+");
	printf("writing PGM to %s in %d*%d\n", filename, width, height);
	fprintf(fd,"%s\n%d %d\n%d\n", "P5", width, height, 255);
	fwrite(buffer, 1, width * height, fd);
	fclose(fd);
    return NULL;
}
