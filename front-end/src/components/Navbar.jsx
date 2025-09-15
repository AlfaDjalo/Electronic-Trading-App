import { useEffect } from "react";
import { Link, NavLink } from "react-router-dom";

export const Navbar = ( {menuOpen, setMenuOpen} ) => {

    useEffect(() => {
        document.body.style.overflow = menuOpen ? "hidden" : ""
    }, [menuOpen])

    return (
        <nav className="fixed top-0 w-full z-40 bg-[rgba(10, 10, 10, 0.8)] backdrop-blur-lg border-b border-white/10 shadow-lg">
            <div className="max-w-5xl mx-auto px-4">
                <div className="flex justify-between items-center h-16">
                    <Link to="/" className="font-mono text-xl font-bold text-white">
                        Logo
                    </Link>

                    <div
                        className="w-7 h-5 relative cursor-pointer z-40 md:hidden"
                        onClick={() => setMenuOpen((prev) => !prev)}
                    >
                        &#9776;
                    </div> 

                    <div className="hidden md:flex items-center space-x-8">
                        <NavLink
                            // to="/"
                            to="/load_data"
                            className={({ isActive }) =>
                                isActive
                                ? "text-white font-semibold"
                                : "text-gray-300 hover:text-white transition-colors"
                            }
                        >
                            Load Data
                        </NavLink>

                        <NavLink
                            to="/view_data"
                            className={({ isActive }) =>
                                isActive
                                ? "text-white font-semibold"
                                : "text-gray-300 hover:text-white transition-colors"
                            }
                        >
                            View Data
                        </NavLink>

                        <NavLink
                            to="/manage_feature_sets"
                            className={({ isActive }) =>
                                isActive
                                ? "text-white font-semibold"
                                : "text-gray-300 hover:text-white transition-colors"
                            }
                        >
                            Manage Feature Sets
                        </NavLink>

                        <NavLink
                            to="/select_model"
                            className={({ isActive }) =>
                                isActive    
                                ? "text-white font-semibold"
                                : "text-gray-300 hover:text-white transition-colors"        
                            }
                        >
                            Select Model
                        </NavLink>

                        <NavLink
                            to="/view_results"
                            className={({ isActive }) =>
                                isActive
                                ? "text-white font-semibold"
                                : "text-gray-300 hover:text-white transition-colors"
                            }
                        >
                            View Results
                        </NavLink>
                    </div>
                </div>
            </div>
        </nav>
    )
}