#include <windows.h>

#include <stdlib.h>
#include <time.h>

#include "ML.h"

// amount of horizontal tiles
#define GRID_HEIGHT 25
// amount of veritical tiles
#define GRID_WIDTH 25

#define GRID_LEN (GRID_HEIGHT * GRID_WIDTH)

// remembers what is drawn on the screen
BYTE SCREEN_GRID[GRID_HEIGHT][GRID_WIDTH];

// remembers what is going on in the game
BYTE GAME_GRID[GRID_HEIGHT][GRID_WIDTH];

#define GAME_STEPS 100
int actionCounter = 0;

typedef struct Step
{
    float *state;
    int output;
    float reward;
} Step;

Network SnakeNN;
Step *snakeSteps[GAME_STEPS];

// height and width of a single tile
#define TILE_SIZE 50
// height in pixels
#define PIXELS_HEIGHT (GRID_HEIGHT * TILE_SIZE)
// width in pixels
#define PIXELS_WIDTH (GRID_WIDTH * TILE_SIZE)

#define GRID_AT(grid, x, y) (grid[y][x])

#define COMP_POINT(p1, p2) (((p1)->x == (p2)->x) && ((p1)->y == (p2)->y))

#define EYE_COLOR ((COLORREF)RGB(0, 0, 0))

typedef enum TILE_RGB
{
    NoneTileRGB = (COLORREF)RGB(0, 0, 0),
    BorderTileRGB = (COLORREF)RGB(255, 255, 255),
    SnakeTileRGB = (COLORREF)RGB(0, 255, 0),
    AppleTileRGB = (COLORREF)RGB(255, 0, 0),
} TileRGB;

typedef enum TILE_TYPE
{
    NoneTile,
    BorderTile,
    SnakeTile,
    AppleTile,
} TileType;

typedef enum SNAKE_DIRECTION
{
    Up,
    Left,
    Down,
    Right,
} SnakeDirections;

typedef struct Point
{
    int x;
    int y;
} Point;
typedef struct POINT_LIST_NODE
{
    struct POINT_LIST_NODE *next;
    Point *point;
} PointList;

Point *ApplePoint;
PointList *SnakePoints;
BYTE LastDirection = 255;
BYTE SnakeDirection = 255;

int GetTileRGB(TileType tileType)
{
    switch (tileType)
    {
    case NoneTile:
        return NoneTileRGB;
    case BorderTile:
        return BorderTileRGB;
    case SnakeTile:
        return SnakeTileRGB;
    case AppleTile:
        return AppleTileRGB;
    default:
        return RGB(128, 128, 128);
    }
}

// random number from low to high including high
// low <= n <= high
int RandomInt(int low, int high)
{
    return (rand() % (high - low + 1)) + low;
}

void FreePointList(PointList *p)
{
    if (!p)
        return;
    if (!p->next)
    {
        GlobalFree(p->point);
        GlobalFree(p);
        return;
    }
    FreePointList(p->next);
    GlobalFree(p->point);
    GlobalFree(p);
}

void NewApple(void)
{
    int randomAppleX = RandomInt(1, GRID_WIDTH - 2);
    int randomAppleY = RandomInt(1, GRID_HEIGHT - 2);
    while (GRID_AT(GAME_GRID, randomAppleX, randomAppleY) == SnakeTile)
    {
        randomAppleX = RandomInt(1, GRID_WIDTH - 2);
        randomAppleY = RandomInt(1, GRID_HEIGHT - 2);
    }
    ApplePoint->x = randomAppleX;
    ApplePoint->y = randomAppleY;
    GRID_AT(GAME_GRID, ApplePoint->x, ApplePoint->y) = AppleTile;
}

void InitializeGame(void)
{
    ZeroMemory(GAME_GRID, sizeof(GAME_GRID));
    ZeroMemory(SCREEN_GRID, sizeof(SCREEN_GRID));

    FreePointList(SnakePoints->next);
    SnakePoints->next = NULL;

    int randomSnakeX = RandomInt(1, GRID_WIDTH - 2);
    int randomSnakeY = RandomInt(1, GRID_HEIGHT - 2);
    int randomAppleX = RandomInt(1, GRID_WIDTH - 2);
    int randomAppleY = RandomInt(1, GRID_HEIGHT - 2);
    while (randomAppleX == randomSnakeX)
    {
        randomAppleX = RandomInt(1, GRID_WIDTH - 2);
    }
    while (randomAppleY == randomSnakeY)
    {
        randomAppleY = RandomInt(1, GRID_HEIGHT - 2);
    }
    for (int y = 0; y < GRID_HEIGHT; y++)
    {
        for (int x = 0; x < GRID_WIDTH; x++)
        {
            GRID_AT(SCREEN_GRID, x, y) = 255;
            if (x == 0 || y == 0 || x == GRID_WIDTH - 1 || y == GRID_HEIGHT - 1)
            {
                GRID_AT(GAME_GRID, x, y) = BorderTile;
            }
        }
    }
    ApplePoint->x = randomAppleX;
    ApplePoint->y = randomAppleY;

    SnakePoints->point->x = randomSnakeX;
    SnakePoints->point->y = randomSnakeY;

    GRID_AT(GAME_GRID, randomAppleX, randomAppleY) = AppleTile;
    GRID_AT(GAME_GRID, randomSnakeX, randomSnakeY) = SnakeTile;
}

void GameOver(HWND hwnd)
{
    // PostMessage(hwnd, WM_QUIT, 0, 0);
    // SnakeDirection = 255;
    // ReinforcementLearning();
    InitializeGame();
    InvalidateRect(hwnd, NULL, FALSE);
    SendMessage(hwnd, WM_PAINT, 0, 0);
}

void GameStep(HWND hwnd)
{
    // Snake has to take a step and update the game grid data

    PointList *newList = (PointList *)GlobalAlloc(GMEM_FIXED, sizeof(*SnakePoints));
    if (!newList)
    {
        return;
    }
    newList->next = NULL;
    newList->point = (Point *)GlobalAlloc(GMEM_FIXED, sizeof(*SnakePoints->point));
    if (!newList->point)
    {
        return;
    }

    switch (SnakeDirection)
    {
    case Up:
        newList->point->x = SnakePoints->point->x;
        newList->point->y = SnakePoints->point->y - 1;
        break;
    case Down:
        newList->point->x = SnakePoints->point->x;
        newList->point->y = SnakePoints->point->y + 1;
        break;
    case Left:
        newList->point->x = SnakePoints->point->x - 1;
        newList->point->y = SnakePoints->point->y;
        break;
    case Right:
        newList->point->x = SnakePoints->point->x + 1;
        newList->point->y = SnakePoints->point->y;
        break;
    }
    LastDirection = SnakeDirection;

    newList->next = SnakePoints;
    SnakePoints = newList;
    newList = NULL;

    // Events to handle:
    //     Snake touches apple
    //     Snake touches border
    //     Snake touches snake
    if (GRID_AT(GAME_GRID, SnakePoints->point->x, SnakePoints->point->y) == BorderTile)
    {
        snakeSteps[actionCounter]->reward = -1.0f;
        GameOver(hwnd);
        return;
    }
    if (GRID_AT(GAME_GRID, SnakePoints->point->x, SnakePoints->point->y) == SnakeTile)
    {
        PointList *last = SnakePoints;
        while (last->next)
            last = last->next;
        if (!(SnakePoints->point->x == last->point->x && SnakePoints->point->y == last->point->y))
        {
            snakeSteps[actionCounter]->reward = -1.0f;
            GameOver(hwnd);
            return;
        }
    }

    if (GRID_AT(GAME_GRID, SnakePoints->point->x, SnakePoints->point->y) == AppleTile)
    {
        snakeSteps[actionCounter]->reward = 1.0f;
        GRID_AT(GAME_GRID, SnakePoints->point->x, SnakePoints->point->y) = SnakeTile;
        NewApple();
    }
    else
    {
        PointList *oneBeforeLast = SnakePoints;
        while (oneBeforeLast->next && oneBeforeLast->next->next)
        {
            oneBeforeLast = oneBeforeLast->next;
        }
        GRID_AT(GAME_GRID, oneBeforeLast->next->point->x, oneBeforeLast->next->point->y) = NoneTile;
        GlobalFree(oneBeforeLast->next);
        oneBeforeLast->next = NULL;
        GRID_AT(GAME_GRID, SnakePoints->point->x, SnakePoints->point->y) = SnakeTile;
    }
    InvalidateRect(hwnd, NULL, FALSE);
    SendMessage(hwnd, WM_PAINT, 0, 0);
}

void ReinforcementLearning()
{
    for (int i = 0; i < GAME_STEPS; i++)
    {
    }
}

int GetSnakeAction()
{
    float *floatGrid = (float *)GlobalAlloc(GMEM_FIXED, sizeof(float) * GRID_LEN);

    for (int i = 0; i < GRID_LEN; i++)
    {
        floatGrid[i] = (float)GAME_GRID[0][i];
    }
    Matrix gameGrid = {
        .rows = 1,
        .cols = GRID_LEN,
        .stride = GRID_LEN,
        .es = floatGrid,
    };
    mat_copy(NETWORK_IN(SnakeNN), gameGrid);
    Network_forward(SnakeNN);

    int answer = 0;
    float ansVal = -1.f;

    for (int i = 0; i < NETWORK_OUT(SnakeNN).cols; i++)
    {
        if (MAT_AT(NETWORK_OUT(SnakeNN), 0, i) > ansVal)
        {
            if ((i == Up && LastDirection == Down) ||
                (i == Down && LastDirection == Up) ||
                (i == Left && LastDirection == Right) ||
                (i == Right && LastDirection == Left))
            {
                // invalid move
            }
            else
            {
                ansVal = MAT_AT(NETWORK_OUT(SnakeNN), 0, i);
                answer = i;
            }
        }
    }
    if (actionCounter < GAME_STEPS)
    {
        for (int i = 0; i < GRID_LEN; i++)
        {
            snakeSteps[actionCounter]->state[i] = floatGrid[i];
        }
        snakeSteps[actionCounter]->output = answer;
    }
    GlobalFree(floatGrid);
    printf("Snake wants:\t%d\n", answer);
    return answer;
}

DWORD WINAPI GameLoop(LPVOID lpParam)
{
    HWND hwnd = (HWND)lpParam;
    while (1)
    {
        if (actionCounter < GAME_STEPS)
        {

            if (SnakeDirection != 255)
            {
                SnakeDirection = GetSnakeAction();
                GameStep(hwnd);
                printf("Reward is:\t%.2f\n\n", snakeSteps[actionCounter]->reward);
                actionCounter++;
                Sleep(2);
            }
        }
        else
        {
            ReinforcementLearning();
            for (int i = 0; i < GAME_STEPS; i++)
            {
                for (int j = 0; j < GRID_LEN; j++)
                {
                    snakeSteps[i]->state[j] = 0;
                }
                snakeSteps[i]->output = 0;
                snakeSteps[i]->reward = 0;
            }
            actionCounter = 0;
        }
    }
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg)
    {
    case WM_KEYDOWN:
        if ((lParam & 0x40000000) == 0)
        {
            switch (wParam)
            {
            case 'W':
                if (LastDirection != Down)
                    SnakeDirection = Up;
                break;
            case 'A':
                if (LastDirection != Right)
                    SnakeDirection = Left;
                break;
            case 'S':
                if (LastDirection != Up)
                    SnakeDirection = Down;
                break;
            case 'D':
                if (LastDirection != Left)
                    SnakeDirection = Right;
                break;
            case VK_SPACE:
                SnakeDirection = 255;
                break;
            default:
                break;
            }
            // Beep(100, 1);
        }
        break;
    case WM_PAINT:
    {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);

        RECT tileRect;
        HBRUSH hBrush;

        // double buffer
        for (int y = 0; y < GRID_HEIGHT; y++)
        {
            for (int x = 0; x < GRID_WIDTH; x++)
            {
                if (GRID_AT(SCREEN_GRID, x, y) != GRID_AT(GAME_GRID, x, y))
                {
                    tileRect.left = (x * TILE_SIZE);
                    tileRect.top = (y * TILE_SIZE);
                    tileRect.right = (tileRect.left + TILE_SIZE);
                    tileRect.bottom = (tileRect.top + TILE_SIZE);
                    TileType tileType = GRID_AT(GAME_GRID, x, y);
                    TileRGB tileRGB = GetTileRGB(tileType);
                    hBrush = CreateSolidBrush(tileRGB);
                    FillRect(hdc, &tileRect, hBrush);
                    DeleteObject(hBrush);
                    GRID_AT(SCREEN_GRID, x, y) = GRID_AT(GAME_GRID, x, y);
                }
            }
        }

        Point *head = SnakePoints->point;
        if (SnakePoints->next)
        {
            Point *lastHead = SnakePoints->next->point;
            tileRect.left = lastHead->x * TILE_SIZE;
            tileRect.top = lastHead->y * TILE_SIZE;
            tileRect.right = tileRect.left + TILE_SIZE;
            tileRect.bottom = tileRect.top + TILE_SIZE;
            hBrush = CreateSolidBrush(SnakeTileRGB);
            FillRect(hdc, &tileRect, hBrush);
            GRID_AT(SCREEN_GRID, lastHead->x, lastHead->y) = SnakeTile;
            DeleteObject(hBrush);
        }
        RECT leftEye;
        RECT rightEye;
        switch (LastDirection)
        {
        case Up:
            leftEye.left = (head->x * TILE_SIZE) + (TILE_SIZE / 4);
            leftEye.top = (head->y * TILE_SIZE) + (TILE_SIZE / 4);

            rightEye.left = (head->x * TILE_SIZE) + (3 * TILE_SIZE / 4);
            rightEye.top = (head->y * TILE_SIZE) + (TILE_SIZE / 4);
            break;
        case Left:
            leftEye.left = (head->x * TILE_SIZE) + (TILE_SIZE / 4);
            leftEye.top = (head->y * TILE_SIZE) + (3 * TILE_SIZE / 4);

            rightEye.left = (head->x * TILE_SIZE) + (TILE_SIZE / 4);
            rightEye.top = (head->y * TILE_SIZE) + (TILE_SIZE / 4);
            break;
        case Down:
            leftEye.left = (head->x * TILE_SIZE) + (3 * TILE_SIZE / 4);
            leftEye.top = (head->y * TILE_SIZE) + (3 * TILE_SIZE / 4);

            rightEye.left = (head->x * TILE_SIZE) + (TILE_SIZE / 4);
            rightEye.top = (head->y * TILE_SIZE) + (3 * TILE_SIZE / 4);
            break;
        case Right:
            leftEye.left = (head->x * TILE_SIZE) + (3 * TILE_SIZE / 4);
            leftEye.top = (head->y * TILE_SIZE) + (TILE_SIZE / 4);

            rightEye.left = (head->x * TILE_SIZE) + (3 * TILE_SIZE / 4);
            rightEye.top = (head->y * TILE_SIZE) + (3 * TILE_SIZE / 4);
            break;
        }
        leftEye.right = leftEye.left + TILE_SIZE / 20;
        leftEye.bottom = leftEye.top + TILE_SIZE / 20;
        rightEye.right = rightEye.left + TILE_SIZE / 20;
        rightEye.bottom = rightEye.top + TILE_SIZE / 20;
        hBrush = CreateSolidBrush(EYE_COLOR);
        FillRect(hdc, &leftEye, hBrush);
        FillRect(hdc, &rightEye, hBrush);
        DeleteObject(hBrush);

        EndPaint(hwnd, &ps);
    }
    break;
    case WM_DESTROY:
        DestroyWindow(hwnd);
        break;
    case WM_QUIT:
        PostQuitMessage(0);
    default:
        return DefWindowProc(hwnd, msg, wParam, lParam);
        break;
    }
}
#include <stdio.h>
void AttachConsoleToWindow()
{
    // Allocate a new console window
    AllocConsole();

    // Redirect standard input, output, and error streams
    freopen("CONOUT$", "w", stdout);
    freopen("CONIN$", "r", stdin);
    freopen("CONOUT$", "w", stderr);
}

const char mainClassName[] = "SnakeWindowClass";
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
    srand((unsigned int)time(NULL));
    AttachConsoleToWindow();

    WNDCLASSEX wc = {
        .hInstance = hInstance,
        .lpszClassName = mainClassName,
        .lpfnWndProc = WndProc,
        .lpszMenuName = NULL,
        .style = 0,
        .hIcon = LoadIcon(NULL, IDI_APPLICATION),
        .hIconSm = LoadIcon(NULL, IDI_APPLICATION),
        .cbSize = sizeof(wc),
        .cbClsExtra = 0,
        .cbWndExtra = 0,
        .hbrBackground = (HBRUSH)(COLOR_WINDOW + 1),
        .hCursor = LoadCursor(hInstance, IDC_ARROW),
    };

    if (!RegisterClassEx(&wc))
    {
        MessageBox(NULL, "Window Registration Failed!", "Error!", MB_ICONEXCLAMATION | MB_OK);
        return 0;
    }

    DWORD windowStyle = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU;

    // calculate the position for window centering
    int screenWidth = GetSystemMetrics(SM_CXSCREEN);
    int screenHeight = GetSystemMetrics(SM_CYSCREEN);
    int startX = (screenWidth - PIXELS_WIDTH) / 2;
    int startY = (screenHeight - PIXELS_HEIGHT) / 2;

    RECT wndRect = {0, 0, PIXELS_WIDTH, PIXELS_HEIGHT};
    AdjustWindowRect(&wndRect, windowStyle, FALSE);

    int windowWidth = wndRect.right - wndRect.left;
    int windowHeight = wndRect.bottom - wndRect.top;

    HWND hwnd = CreateWindowEx(
        0,
        mainClassName,
        "Snake",
        windowStyle,
        startX, startY, windowWidth, windowHeight,
        NULL, NULL, hInstance, NULL);
    if (!hwnd)
    {
        MessageBox(NULL, "Window Creation Failed!", "Error!", MB_ICONEXCLAMATION | MB_OK);
        return 0;
    }

    SnakePoints = (PointList *)GlobalAlloc(GMEM_FIXED, sizeof(*SnakePoints));
    SnakePoints->next = NULL;
    SnakePoints->point = (Point *)GlobalAlloc(GMEM_FIXED, sizeof(*SnakePoints->point));
    ApplePoint = (Point *)GlobalAlloc(GMEM_FIXED, sizeof(*ApplePoint));
    for (int i = 0; i < GAME_STEPS; i++)
    {
        snakeSteps[i] = (Step *)GlobalAlloc(GMEM_FIXED, sizeof(*snakeSteps[i]));
        snakeSteps[i]->state = (float *)GlobalAlloc(GMEM_FIXED, GRID_LEN * sizeof(*snakeSteps[i]->state));
        snakeSteps[i]->output = 0;
        snakeSteps[i]->reward = 0.f;
    }
    InitializeGame();

    HANDLE hThread;
    DWORD threadID;
    hThread = CreateThread(
        NULL,
        0,
        GameLoop,
        hwnd,
        0,
        &threadID);
    if (!hThread)
    {
        MessageBox(hwnd, "The program failed to create a thread", "Fatal Error", MB_OK | MB_ICONERROR);
        return 0;
    }

    size_t layers[] = {GRID_LEN, 100, 4};
    size_t len = sizeof(layers) / sizeof(*layers);
    ActivationType acts[] = {SIGMOID, SIGMOID};
    SnakeNN = NeuralNetwork(layers, len, NULL);
    Network_rand(SnakeNN, -25, 25);

    ShowWindow(hwnd, nCmdShow);
    UpdateWindow(hwnd);

    MSG msg;
    while (GetMessage(&msg, hwnd, 0, 0) > 0)
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    CloseHandle(hThread);
    return msg.wParam;
}