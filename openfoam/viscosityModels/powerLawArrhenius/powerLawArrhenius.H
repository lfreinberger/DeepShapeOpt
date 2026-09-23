/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2016 OpenFOAM Foundation
    Copyright (C) 2017-2019 OpenCFD Ltd.
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Class
    Foam::viscosityModels::powerLawArrhenius   // CHANGED: neuer Klassenname

Description
    Power-law Modell mit zusätzlichem Arrhenius-Faktor in 1/T:
        nu = k * exp(Eactive_/(Rconst_*T)) * |gammaDot|^(n-1)
    mit Eactive_ [J/mol] und Rconst_ [J/(mol·K)].
               

SourceFiles
    powerLawArrhenius.C                            

\*---------------------------------------------------------------------------*/

#ifndef viscosityModels_powerLawArrhenius_H
#define viscosityModels_powerLawArrhenius_H

#include "viscosityModel.H"
#include "dimensionedScalar.H"
#include "volFields.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace viscosityModels
{

/*---------------------------------------------------------------------------*\
                           Class powerLawArrhenius Declaration
\*---------------------------------------------------------------------------*/

class powerLawArrhenius
:
    public viscosityModel
{
    // Private data

        dictionary powerLawArrheniusCoeffs_;

        dimensionedScalar k_;
        dimensionedScalar n_;
        dimensionedScalar nuMin_;
        dimensionedScalar nuMax_;
        
        dimensionedScalar Eactive_;
        dimensionedScalar Rconst_;

        //- Under-relaxation factor for the viscosity update (1 = none).
        //  Blends the newly computed nu with the previous value each
        //  correct() call: nu = (1-w)*nu_old + w*calcNu(). Strongly
        //  shear-thinning melts (large nuMax/nuMin ratio, small n) often need
        //  w < 1 for the SIMPLE loop to stay stable.
        scalar viscosityRelaxation_;



protected:

    // Protected data

        volScalarField nu_;


    // Protected Member Functions

        //- Calculate and return the laminar viscosity
        tmp<volScalarField> calcNu() const;


public:

    //- Runtime type information
    TypeName("powerLawArrhenius");


    // Constructors

        //- Construct from components
        powerLawArrhenius
        (
            const word& name,
            const dictionary& viscosityProperties,
            const volVectorField& U,
            const surfaceScalarField& phi
        );


    //- Destructor
    virtual ~powerLawArrhenius() = default;


    // Member Functions

        //- Return the laminar viscosity
        virtual tmp<volScalarField> nu() const
        {
            return nu_;
        }

        //- Return the laminar viscosity for patch
        virtual tmp<scalarField> nu(const label patchi) const
        {
            return nu_.boundaryField()[patchi];
        }

        //- Correct the laminar viscosity (optionally under-relaxed)
        virtual void correct()
        {
            if (viscosityRelaxation_ >= 1)
            {
                nu_ = calcNu();
            }
            else
            {
                nu_ = (1 - viscosityRelaxation_)*nu_
                    + viscosityRelaxation_*calcNu();
            }
        }

        //- Read transportProperties dictionary
        virtual bool read(const dictionary& viscosityProperties);
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace viscosityModels
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
