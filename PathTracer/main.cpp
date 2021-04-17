#include <iostream>

#include "Program.h"

constexpr int WINDOW_WIDTH  = 1920 / 2, WINDOW_HEIGHT = 1080 / 2;

int main() {
	Program::Initialize(WINDOW_WIDTH, WINDOW_HEIGHT);

	// Main loop.
	while (!Program::ShouldTerminate()) {
		Program::Update();
		Program::Draw();
	}

	Program::Terminate();
	return 0;
}
