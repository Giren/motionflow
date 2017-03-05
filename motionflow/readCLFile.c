//
//  readCLFile.c
//  hello
//
//  Created by Michael Nienhaus on 24.04.15.
//
//

#include "readCLFile.h"
#include <stdio.h>
#include <stdlib.h>


const char *ksrc(const char *filename) {
    char * buffer;
    long lSize;
    FILE * fp;
    
    fp = fopen (filename, "rb" );
    if( !fp ) perror(filename),exit(1);
    
    //get filesize
    fseek(fp , 0L , SEEK_END);
    lSize = ftell(fp);
    rewind(fp);
    
    buffer = (char *)calloc(1, lSize+1);
    if(!buffer) fclose(fp),fputs("memory alloc fails",stderr),exit(1);
    
    if( 1!=fread( buffer , lSize, 1 , fp) ){
        fclose(fp);
        free(buffer);
        fputs("entire read fails", stderr), exit(1);
    }
        //fclose(fp),free(buffer),fputs("entire read fails",stderr),exit(1);
    
    fclose(fp);
    return buffer;;
}