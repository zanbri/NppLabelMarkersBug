#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <ImageIO.h>
#include <Exceptions.h>

#include <string.h>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include <npp.h>

#include <helper_string.h>
#include <helper_cuda.h>

int main(int argc, char *argv[])
{
    try {
        std::string sFilename = "test.pgm";
        char *filePath;

        int dev = 0;

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        std::cerr << "cudaSetDevice GPU" << dev << " = " << deviceProp.name << std::endl;

        // if we specify the filename at the command line, then we only test sFilename[0].
        int file_errors = 0;
        std::ifstream infile(sFilename.data(), std::ifstream::in);

        if (infile.good()) {
            std::cout << "File opened: <" << sFilename.data() << "> successfully!" << std::endl;
            file_errors = 0;
            infile.close();
        } else {
            std::cout << "Unable to open file: <" << sFilename.data() << ">" << std::endl;
            file_errors++;
            infile.close();
        }

        if (file_errors > 0) {
            exit(EXIT_FAILURE);
        }

        std::string sResultFilename = sFilename;

        std::string::size_type dot = sResultFilename.rfind('.');

        if (dot != std::string::npos) {
            sResultFilename = sResultFilename.substr(0, dot);
        }

        sResultFilename += "_connected_components.pgm";

        // declare a host image object for an 8-bit grayscale image
        npp::ImageCPU_8u_C1 oHostSrc;

        // load gray-scale image from disk
        npp::loadImage(sFilename, oHostSrc);

        // declare a device image and copy construct from the host image,
        // i.e. upload host to device
        npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);

        NppiSize oSrcSize = {(int)oDeviceSrc.width() , (int)oDeviceSrc.height()};
        // allocate device image of appropriately reduced size
        npp::ImageNPP_8u_C1 oDeviceDst8u(oSrcSize.width, oSrcSize.height);
        npp::ImageNPP_32s_C1 oDeviceDst32u(oSrcSize.width, oSrcSize.height);

        int nBufferSize = 0;
        Npp8u * pScratchBufferNPP = 0;

        // get necessary scratch buffer size and allocate that much device memory
        NPP_CHECK_NPP(
                nppiLabelMarkersUFGetBufferSize_32u_C1R(oSrcSize, &nBufferSize)
        );

        cudaMalloc((void **)&pScratchBufferNPP, nBufferSize);

        // Now generate label markers using 8 way search mode (nppiNormInf).
        if ((nBufferSize > 0) && (pScratchBufferNPP != 0)) {
            NPP_CHECK_NPP(
                    nppiLabelMarkersUF_8u32u_C1R(
                        oDeviceSrc.data(), oDeviceSrc.pitch(),
                        reinterpret_cast<Npp32u *>(oDeviceDst32u.data()), oDeviceDst32u.pitch(),
                        oSrcSize, nppiNormInf, pScratchBufferNPP)
            );
        }


        // free scratch buffer memory
        cudaFree(pScratchBufferNPP);

        // Compress the generated list of labels to fit into 8 bits.
        // 
        // Get necessary scratch buffer size and allocate that much device memory
        //
        // NOTE: Since the image was generated by the nppiLabelMarkersUF functions, I am providing (ROI width * ROI height) as the starting number
        int maxLabel = oSrcSize.width * oSrcSize.height;
        NPP_CHECK_NPP(
                nppiCompressMarkerLabelsGetBufferSize_32u8u_C1R(maxLabel, &nBufferSize)
        );

        cudaMalloc((void **)&pScratchBufferNPP, nBufferSize);

        if ((nBufferSize > 0) && (pScratchBufferNPP != 0)) {

            NPP_CHECK_NPP(
                    nppiCompressMarkerLabels_32u8u_C1R(
                        reinterpret_cast<Npp32u *>(oDeviceDst32u.data()), oDeviceDst32u.pitch(),
                        oDeviceDst8u.data(), oDeviceDst8u.pitch(),
                        oSrcSize, maxLabel, &maxLabel, pScratchBufferNPP)
            );

        }

        // free scratch buffer memory
        cudaFree(pScratchBufferNPP);

        std::cout << "Number of connected components: " << maxLabel << std::endl;

        // Declare a host image for the result
        npp::ImageCPU_8u_C1 oHostDst8u(oDeviceDst8u.size());
        // and copy the device result data into it
        oDeviceDst8u.copyTo(oHostDst8u.data(), oHostDst8u.pitch());

        saveImage(sResultFilename, oHostDst8u);
        std::cout << "Saved image: " << sResultFilename << std::endl;

        nppiFree(oDeviceSrc.data());
        nppiFree(oDeviceDst32u.data());
        nppiFree(oDeviceDst8u.data());

        exit(EXIT_SUCCESS);
    }
    catch (npp::Exception &rException)
    {
        std::cerr << "Program error! The following exception occurred: \n";
        std::cerr << rException << std::endl;
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
    }
    catch (...)
    {
        std::cerr << "Program error! An unknow type of exception occurred. \n";
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
        return -1;
    }

    return 0;
}