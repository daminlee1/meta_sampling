#include <Python.h>
#include <stdio.h>
#include "aes.h"

int init(void)
{
  Py_Initialize();
  if (!Py_IsInitialized()) {
    return 0;
  }
  return 1;
}

void finalize(void)
{
  Py_Finalize();
}

void run_main(const char* filename)
{
  FILE* fp = fopen(filename, "rb");
  if (!fp) {
    return;
  }

  fseek(fp, 0, SEEK_END);
  long int fsize = ftell(fp);
  unsigned char* encrypted_code = (unsigned char*)malloc((fsize + 1) * sizeof(unsigned char));
  unsigned char* decrypted_code = (unsigned char*)malloc((fsize + 1) * sizeof(unsigned char));
  char key[] = "8765432187654321";
  rewind(fp);
  fread(encrypted_code, 1, fsize, fp);
  fclose(fp);
  aes_decrypt(encrypted_code, (unsigned char*)key, decrypted_code, fsize);
  long int len = *((long int*)decrypted_code);
  decrypted_code[sizeof(long int) + len] = 0;
  PyRun_SimpleString(decrypted_code + sizeof(long int));
  free(encrypted_code);
  free(decrypted_code);
}

int main(int argc, char** argv)
{
  if (argc != 2) {
    printf("Wrong argument\n");
    exit(-1);
  }

  if (init()) {
    run_main(argv[1]);
    finalize();
  }

  return 0;
}
