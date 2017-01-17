#include <vector>
#include <math.h>
#include <stdio.h>
using namespace std;

void HaarDWT(std::vector<float> &signal)
{
    for (int i = signal.size()>>1;;i >>= 1) {
        for (int j = 0; j < i; ++j) {
            float s0 = (signal[2*j]+signal[2*j+1])/2;
            float s1 = signal[2*j]-s0;
            signal[2*j] = s0;
            signal[2*j+1]=s1;
            if(i == 1)
                return;
        }
    }
}

void IHaarDWT(std::vector<float> &tsignal)
{
    for (int i = tsignal.size()>>1;;i >>= 1) {
        for (int j = 0; j < i; ++j) {
            float s0 = tsignal[2*j]+tsignal[2*j+1];
            float s1 = tsignal[2*j]-tsignal[2*j+1];
            tsignal[2*j] = s0;
            tsignal[2*j+1]=s1;
            if(i == 1)
                return;
        }
    }
}

int main(int argc, char *argv[])
{

    std::vector<float> signal = {8,4,1,3}; //c++11 //signal has length 2^n
    HaarDWT(signal);
    for(auto& ob : signal)
        printf("%f ",ob);
    IHaarDWT(signal);
    printf("\n");
    for(auto& ob : signal)
        printf("%f ",ob);
    return 0;
}

