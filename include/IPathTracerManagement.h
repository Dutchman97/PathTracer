#include "IPathTracer.h"

#include <Windows.h>
#include <cstdlib>
#include <iostream>

#define DLL_FAIL(desc, libraryName) { DWORD errorCode = GetLastError(); std::cout << desc << " (library '" << libraryName << "', error '" << errorCode << "')" << std::endl; return false; }

bool LoadPathTracerLibrary(const char* libraryName, CreatePathTracerFunc* createFuncPtr, DestroyPathTracerFunc* destroyFuncPtr) {
	HINSTANCE hinstLibrary = LoadLibraryA(libraryName);

	if (hinstLibrary) {
		CreatePathTracerFunc Create = (CreatePathTracerFunc)GetProcAddress(hinstLibrary, "Create");
		DestroyPathTracerFunc Destroy = (DestroyPathTracerFunc)GetProcAddress(hinstLibrary, "Destroy");

		if (Create && Destroy) {
			*createFuncPtr = Create;
			*destroyFuncPtr = Destroy;
			return true;
		}
		else {
			DLL_FAIL("Unable to get Create and Destroy function addresses.", libraryName);
		}
	}
	else {
		DLL_FAIL("Unable to load library.", libraryName);
	}
	return false;
}

bool UnloadPathTracerLibrary(const char* libraryName) {
	HINSTANCE hinstLibrary = GetModuleHandleA(libraryName);

	if (hinstLibrary) {
		BOOL success = FreeLibrary(hinstLibrary);
		if (success) {
			return true;
		}
		else {
			DLL_FAIL("Unable to free library.", libraryName);
		}
	}
	else {
		DLL_FAIL("Unable to get handle of module.", libraryName);
	}
	return false;
}